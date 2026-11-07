//! An idle engine's gather window (`Engine::with_gather_window`) holds only while other requests are still arriving
//! (`Observability::arriving`), plus a few milliseconds of grace after the last one seen (`runtime::GATHER_GRACE`): a request that wakes
//! the idle engine alone starts at once (after the grace); one that wakes it while another is in the HTTP layer waits for that one's
//! submit, and no longer; a request refused before it submits ends the wait too. Timing only, never a token.
mod common;

use common::*;
use std::io::Write;
use std::net::TcpStream;
use std::time::{Duration, Instant};

const WINDOW: Duration = Duration::from_secs(3);

fn server() -> Server {
    start(Opts { gather: WINDOW, ..Default::default() })
}

fn body(max_tokens: u64) -> serde_json::Value {
    chat_body(serde_json::json!({"max_tokens": max_tokens}))
}

/// A request whose head (and `Content-Length`) is sent but not its body: it has entered the middleware and arrives until the body ends.
fn held(port: u16, body: &[u8]) -> TcpStream {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).unwrap();
    s
}

fn arriving(s: &Server) -> u64 {
    s.metric(|m| &m.observability.arriving)
}

#[test]
fn a_request_that_arrives_alone_does_not_wait_the_gather_window() {
    let s = server();
    let t = Instant::now();
    let (status, text) = http(s.port, "POST", "/v1/chat/completions", Some(&body(2)));
    assert_eq!(status, 200, "{text}");
    // mutation: the old fixed window (wait `WINDOW` whenever a request wakes the idle engine) and this takes 3 s
    assert!(t.elapsed() < WINDOW / 2, "a lone request waited {:?}", t.elapsed());
    assert_eq!(arriving(&s), 0, "a submitted request no longer arrives");
}

#[test]
fn a_request_waits_for_one_still_arriving_and_only_until_it_submits() {
    let s = server();
    let late = body(2).to_string().into_bytes();
    let mut b = held(s.port, &late);
    s.wait(Duration::from_secs(5), "the held request to enter the middleware", |s| arriving(s) == 1);
    let port = s.port;
    let t = Instant::now();
    let a = std::thread::spawn(move || {
        let r = http(port, "POST", "/v1/chat/completions", Some(&body(2)));
        (r, Instant::now())
    });
    // the first request wakes the engine, which waits: the held one is still arriving
    std::thread::sleep(Duration::from_millis(400));
    assert!(!a.is_finished(), "the first request did not wait for the one still arriving");
    b.write_all(&late).unwrap();
    let ((status, text), done) = a.join().unwrap();
    assert_eq!(status, 200, "{text}");
    let waited = done.duration_since(t);
    assert!(waited >= Duration::from_millis(400), "{waited:?}");
    // the wait ended at the held request's submit, not at the end of the window
    assert!(waited < WINDOW - Duration::from_millis(500), "the wait ran to the end of the window: {waited:?}");
    let mut raw = Vec::new();
    use std::io::Read;
    b.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let _ = b.read_to_end(&mut raw);
    assert!(String::from_utf8_lossy(&raw).starts_with("HTTP/1.1 200"), "{}", String::from_utf8_lossy(&raw));
    assert_eq!(arriving(&s), 0);
}

#[test]
fn a_request_refused_before_it_submits_ends_the_wait() {
    let s = server();
    let bad = b"{ not json".to_vec();
    let mut b = held(s.port, &bad);
    s.wait(Duration::from_secs(5), "the held request to enter the middleware", |s| arriving(s) == 1);
    let port = s.port;
    let t = Instant::now();
    let a = std::thread::spawn(move || http(port, "POST", "/v1/chat/completions", Some(&body(2))));
    std::thread::sleep(Duration::from_millis(400));
    b.write_all(&bad).unwrap(); // refused while it parses: it never submits
    let (status, text) = a.join().unwrap();
    assert_eq!(status, 200, "{text}");
    assert!(t.elapsed() < WINDOW - Duration::from_millis(500), "the wait ran to the end of the window: {:?}", t.elapsed());
    s.wait(Duration::from_secs(5), "the refused request to be settled", |s| arriving(s) == 0);
}

#[test]
fn a_client_that_goes_away_before_its_body_ends_is_settled() {
    let s = server();
    let b = held(s.port, &body(2).to_string().into_bytes());
    s.wait(Duration::from_secs(5), "the held request to enter the middleware", |s| arriving(s) == 1);
    drop(b);
    s.wait(Duration::from_secs(5), "the dropped request to be settled", |s| arriving(s) == 0);
    let t = Instant::now();
    let (status, text) = http(s.port, "POST", "/v1/chat/completions", Some(&body(2)));
    assert_eq!(status, 200, "{text}");
    assert!(t.elapsed() < WINDOW / 2, "a lone request waited {:?}", t.elapsed());
}

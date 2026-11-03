//! Local fixtures only. Consume complete HTTP request bodies, never assume
//! headers and body arrive in the same TCP read.
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};
use url::Url;
#[derive(Debug)]
pub(crate) struct RecordedRequest {
    pub head: String,
    pub body: Vec<u8>,
}
pub(crate) async fn read_request(stream: &mut TcpStream) -> RecordedRequest {
    let mut request = Vec::new();
    loop {
        let mut chunk = [0; 4096];
        let n = stream.read(&mut chunk).await.unwrap();
        assert_ne!(n, 0, "fixture request ended early");
        request.extend_from_slice(&chunk[..n]);
        assert!(request.len() <= 128 * 1024);
        if let Some(split) = request.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8(request[..split].to_vec()).unwrap();
            let length: usize = head
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().unwrap())
                })
                .unwrap_or(0);
            if request.len() >= split + 4 + length {
                return RecordedRequest {
                    head,
                    body: request[split + 4..split + 4 + length].to_vec(),
                };
            }
        }
    }
}
pub(crate) async fn serve(
    responses: Vec<(u16, String)>,
) -> (Url, JoinHandle<Vec<RecordedRequest>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
    let server = tokio::spawn(async move {
        let mut requests = vec![];
        for (status, body) in responses {
            let (mut stream, _) = listener.accept().await.unwrap();
            requests.push(read_request(&mut stream).await);
            let response = format!(
                "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        }
        requests
    });
    (url, server)
}
pub(crate) fn config(token_url: Url) -> crate::connector::oauth::OAuthProviderConfig {
    let mut config = crate::connector::oauth::OAuthProviderConfig::new(
        Url::parse("https://provider.example/authorize").unwrap(),
        token_url,
        "test-client",
    );
    config.allow_http_loopback = true;
    config
}

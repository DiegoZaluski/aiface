//! Shared test fixtures for the oauth module (and its provider adapters).
//!
//! Local loopback HTTP fixtures only: every server binds `127.0.0.1:0` and
//! consumes complete request bodies instead of assuming headers and body
//! arrive in the same TCP read.

pub(crate) mod support;

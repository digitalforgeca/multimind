//! Signal store implementations.
//!
//! Built-in stores:
//! - [`postgres::PgSignalStore`] — PostgreSQL (feature `postgres`)
//! - [`sqlite::SqliteSignalStore`] — SQLite (feature `sqlite`)
//!
//! Custom stores can implement [`SignalStore`](crate::SignalStore) directly.

#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(feature = "sqlite")]
pub mod sqlite;

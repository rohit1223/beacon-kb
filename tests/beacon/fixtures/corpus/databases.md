# SQL Databases and Indexing

Relational databases store data in tables with rows and columns.
SQL, or Structured Query Language, is the standard language for querying and manipulating relational data.
Indexes are data structures that speed up lookups by maintaining a sorted copy of one or more columns.
A B-tree index allows the database to find rows matching a WHERE clause without scanning every row.
Without an index, the engine performs a full table scan, which is slow on large tables.

Transactions group multiple SQL statements into an atomic unit of work.
ACID stands for Atomicity, Consistency, Isolation, and Durability.
Atomicity guarantees all statements in a transaction succeed or all are rolled back.
Consistency ensures the database moves from one valid state to another.
Isolation controls how concurrent transactions see each other's intermediate changes.
Durability guarantees that committed changes survive crashes.

Common SQL databases include PostgreSQL, MySQL, and SQLite.
PostgreSQL supports advanced features like window functions, CTEs, and JSONB columns.
Proper indexing strategy is critical for query performance in production systems.

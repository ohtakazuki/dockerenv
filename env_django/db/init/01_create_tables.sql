create table book
(
  id serial NOT NULL PRIMARY KEY,
  title text,
  insert_timestamp timestamp with time zone
);

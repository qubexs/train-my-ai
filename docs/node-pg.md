# Node.js + PostgreSQL example (pg)
To connect Node.js to PostgreSQL, use the pg package:

```js
import pg from 'pg';
const pool = new pg.Pool({ connectionString: 'postgres://user:pass@localhost:5432/mydb' });
const { rows } = await pool.query('SELECT $1::int AS tambah', [1 + 2]);
console.log(rows); // [{ tambah: 3 }]
```

For MySQL, use mysql2/promise with createPool().

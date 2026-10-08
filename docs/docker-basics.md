# Docker basics untuk pembangun
Perintah harian:

```bash
docker ps                 # kontena berjalan
docker images             # imej tempatan
docker build -t app:1 .   # bina imej
docker run -d -p 3000:3000 app:1
docker logs -f <nama>     # ikut log
docker stop <nama>        # henti
```

Dockerfile Node minima:
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
CMD ["node", "server.js"]
```

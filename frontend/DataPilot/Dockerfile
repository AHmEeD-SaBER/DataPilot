FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

RUN npm install --save-dev @angular/cli

COPY . .

RUN sed -i 's/"outputMode": "server"/"outputMode": "static"/g' angular.json

RUN ./node_modules/.bin/ng build --no-ssr

RUN rm -rf /app/dist/DataPilot/browser/upload \
           /app/dist/DataPilot/browser/results \
           /app/dist/DataPilot/browser/train \
           /app/dist/DataPilot/browser/index.html

RUN mv /app/dist/DataPilot/browser/index.csr.html \
       /app/dist/DataPilot/browser/index.html

FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist/DataPilot/browser /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
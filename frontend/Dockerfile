# Front end: build the React app, serve it with nginx, and proxy /api to
# the backend - so the browser sees one origin and the session cookie is
# an ordinary first-party cookie, exactly as with the Vite dev proxy.

FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ENV VITE_API_BASE=/api
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 8080

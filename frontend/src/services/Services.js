import axios from "axios";

const api_posts = axios.create({
  baseURL: "http://127.0.0.1:5000",
  withCredentials: false,
  headers: {
    Accept: "application/json",
    "Content-Type": "application/json",
  },
});

export default {
  getimagenes() {
    return api_posts.get("/imagen");
  },
  postimagen(post) {
    return api_posts.post("/posts", post);
  },
};

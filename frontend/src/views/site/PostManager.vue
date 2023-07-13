<template>
  <div class="Posts">
    <h1></h1>
    <Posts />
  </div>
</template>

<script>
// @ is an alias to /src
import Posts from "@/components/Posts.vue";

export default {
  name: "Post",
  components: {
    Posts,
  },
};
</script>

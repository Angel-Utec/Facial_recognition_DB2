<template>
  <div>
    <h1></h1>
    <Categories />
  </div>
</template>

<script>
// @ is an alias to /src
import Categories from "@/components/Searchs.vue";

export default {
  name: "search",
  components: {
    Categories,
  },
};
</script>

<template>
  <div class="Home">
    <h1></h1>
    <Homes />
  </div>
</template>

<script>
// @ is an alias to /src
import Homes from "@/components/Homes.vue";

export default {
  name: "Home",
  components: {
    Homes,
  },
};
</script>

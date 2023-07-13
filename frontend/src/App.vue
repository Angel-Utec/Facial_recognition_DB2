<template>
  <div id="background-container">
    <HeaderBar>
      <nav class="custom-navbar">
        <div>
          <a href="" class="brand-logo right">Proyecto base de datos 2 </a>
          <ul id="nav-mobile" class="left hide-on-med-and-down">
            <li><router-link to="/">Home</router-link></li>
            <li><router-link to="/search">Search</router-link></li>
          </ul>
        </div>
      </nav>
    </HeaderBar>
    <div id="content-container">
      <div class="content">
        <router-view @logout="onLogout"></router-view>
      </div>
    </div>
    <div id="background-gif">
      <img src="./assets/ado_cloud.gif" alt="Background GIF" />
    </div>
  </div>
</template>

<style>
body {
  overflow: hidden;
  margin: 0;
}

#background-container {
  position: relative;
  overflow: auto;
  height: 100vh;
}

#content-container {
  position: relative;
  z-index: 1;
  height: 100%;
}

#background-gif {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: -1; /* Asegura que el GIF esté en el fondo */
  pointer-events: none; /* Evita que el GIF interfiera con los elementos de primer plano */
}

.custom-navbar {
  height: 75px;
  padding: 0px;
  background-color: green;
}

.logout-button {
  background: none;
  border: none;
  color: inherit;
  font: inherit;
  cursor: pointer;
  text-decoration: underline;
}
</style>

<script>
import M from "materialize-css";

export default {
  data() {
    return {
      user_logeado: false,
    };
  },
  created() {
    this.logeado();
  },
  mounted() {
    M.AutoInit();
  },
  methods: {
    logeado() {
      const storedUserID = localStorage.getItem("user_id");
      this.user_logeado = !!storedUserID;
    },
    deslogearte() {
      localStorage.removeItem("user_id");
      this.user_logeado = false;
      this.$router.push("/login"); // Redirige al componente de inicio de sesión después de desloguearse
    },
    onLogout() {
      this.deslogearte();
    },
  },
};
</script>

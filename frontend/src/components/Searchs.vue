<template>
  <div>
    <div class="container section">
      <div class="row card-panel">
        <h4>
          <center>Ingresar archivos</center>
        </h4>
        <form class="col s12" @submit.prevent="SubirImagen">
          <div class="file-field input-field">
            <div class="btn">
              <span>File</span>
              <input type="file" accept="image/*" @change="handleFileUpload" />
            </div>
            <div class="file-path-wrapper">
              <input
                class="file-path validate"
                type="text"
                placeholder="Sube tu imagen"
              />
            </div>
          </div>
          <div class="row">
            <div class="input-field col s12">
              <i class="material-icons prefix">clear_all</i>
              <input
                v-model="busqueda.top_k"
                type="text"
                class="validate"
                required
              />
              <label for="Ingresa la categoria">Ingresa el top K</label>
            </div>
          </div>
          <button
            class="btn waves-effect waves-green right"
            type="submit"
            name="enviar"
          >
            Enviar
            <i class="material-icons right">send</i>
          </button>
        </form>
      </div>
    </div>
    <div class="container section">
      <h4>
        Vistas de las coincidencias del menos al mas parecido usando el
        Algoritmo del KNN sequential
      </h4>
      <!--horizontal scrolling cards    -->
      <div class="row-b">
        <div
          v-for="imagen in datos"
          :key="imagen.id"
          class="card-b card sticky-action large"
        >
          <div class="card-image waves-effect waves-block waves-light">
            <img
              class="activator"
              :src="
                require('@/assets/' + imagen.nombre + '/' + imagen.foto_espec)
              "
            />
          </div>
          <div class="card-content">
            <span class="card-title activator grey-text text-darken-4"
              >{{ imagen.nombre
              }}<i class="material-icons right">more_vert</i></span
            >
          </div>
          <div class="card-reveal">
            <span class="card-title grey-text text-darken-4"
              >Datos<i class="material-icons right">close</i></span
            >
            <p>Distancia: {{ imagen.distancia }}</p>
            <p>Nombre: {{ imagen.nombre }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import Services from "@/services/Services";

export default {
  name: "Category-manager",

  data() {
    return {
      busqueda: {
        busqueda_input: "",
        top_k: "",
      },
      datos: [],
    };
  },

  async created() {
    await this.getImagenes();
  },
  methods: {
    async getImagenes() {
      try {
        const response = await Services.getimagenes();
        this.datos = response.data.imagenes;
        console.log(this.datos);
      } catch (error) {
        console.log(error);
      }
    },
    handleFileUpload(event) {
      this.archivoImagen = event.target.files[0];
    },
    SubirImagen() {
      if (this.archivoImagen) {
        const formData = new FormData();
        formData.append("imagen", this.archivoImagen);
        formData.append("top_k", this.busqueda.top_k);
        Services.postimagen(formData).then((response) => {
          this.datos = response.data.posts;
        });
      } else {
        console.log("No se ha seleccionado ningún archivo.");
      }
    },
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style>
.card-b {
  display: inline-block;
  margin: 1rem;
  width: 30rem;
  height: 20rem;
  position: relative;
  overflow: hidden;
}

.row-b {
  overflow-x: auto;
  white-space: nowrap;
}

.title-bg {
  background-color: rgb(187, 182, 182); /* Color de fondo blanco */
  padding: 10px; /* Espaciado interno */
}

.tables-wrapper {
  display: flex;
  justify-content: space-between;
}

.row.card-panel.low-opacity {
  background-color: rgba(
    255,
    255,
    255,
    0.6
  ); /* Color de fondo con baja opacidad */
}
</style>

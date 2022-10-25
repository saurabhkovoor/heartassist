import Vue from 'vue'
import App from './App.vue'
import router from "./routes.js"

// createApp(App).mount('#app')

Vue.config.productionTip = false

new Vue({
    router,
    render: h=> h(App),
}).$mount("#app")
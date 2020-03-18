
const PlotlyChart = {
    props: ['graph'],
    render(h) {
        return h('div', {
            attrs: {
                id: this.graph.id,
            }
        })
    },

    mounted() {
        this.$nextTick(function () {
                let x = JSON.parse(this.graph.json)
                Plotly.react(this.graph.id, x.data, x.layout, {responsive: true});
            }
        )
    },
};


var vm = new Vue({
    el: '#app',

    components: {
        'plotly-chart': PlotlyChart,
    },

    data() {
        return {
            version: 'Version information unavailable',
            history: [],
            historyIdx: 0,
            sim_pars: {},
            epi_pars: {},
            graphs: [],
            running: false,
            err: '',
            reset_options: ['Example', 'Seattle'], // , 'Wuhan', 'Global'],
            reset_choice: 'Example'
        }
    },

    async created() {
        this.get_version();
        this.resetpars('Example');
    },

    filters: {
        to2sf(value) {
            return Number(value).toFixed(2)
        }
    },

    methods: {

        async get_version() {
            let response = await sciris.rpc('get_version');
            vm.version = response.data
        },

        async runSim() {
            vm.running = true;
            vm.graphs = [];

            console.log(vm.status);
            console.log(vm.sim_pars, vm.epi_pars);

            // Run a a single sim
            try{
                let response = await sciris.rpc('plot_sim', [vm.sim_pars, vm.epi_pars]);
                vm.graphs = response.data.graphs;
                vm.err = response.data.err;
                vm.sim_pars= response.data.sim_pars;
                vm.epi_pars = response.data.epi_pars;
                vm.history.push(JSON.parse(JSON.stringify({sim_pars:vm.sim_pars, epi_pars:vm.epi_pars, graphs:vm.graphs})));
                vm.historyIdx = vm.history.length-1;

            } catch (e) {
                vm.err = 'Error running model: ' + e;
            }
            vm.running = false;


        },

        async resetpars(reset_choice) {
            let response = await sciris.rpc('get_defaults', [reset_choice]);
            vm.sim_pars = response.data.sim_pars;
            vm.epi_pars = response.data.epi_pars;
            vm.graphs = [];
        },

        loadPars(){
            vm.sim_pars = vm.history[vm.historyIdx].sim_pars;
            vm.epi_pars = vm.history[vm.historyIdx].epi_pars;
            vm.graphs = vm.history[vm.historyIdx].graphs;
        }
    }
})

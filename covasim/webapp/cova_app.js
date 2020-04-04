const PlotlyChart = {
    props: ['graph'],
    render(h) {
        return h('div', {
            attrs: {
                id: this.graph.id,
            }
        });
    },

    mounted() {
        this.$nextTick(function () {
            let x = JSON.parse(this.graph.json);
            x.responsive = true;
            Plotly.react(this.graph.id, x);
        }
        );
    },
};


var vm = new Vue({
    el: '#app',

    components: {
        'plotly-chart': PlotlyChart,
    },

    data() {
        return {
            version: 'Unable to connect to server!', // This text will display instead of the version
            history: [],
            historyIdx: 0,
            sim_pars: {},
            epi_pars: {},
            show_animation: false,
            result: { // Store currently displayed results
                graphs: [],
                summary: {},
                files: {},
            },
            paramError: {},
            running: false,
            err: '',
            reset_options: ['Example', 'Seattle'], // , 'Wuhan', 'Global'],
            reset_choice: 'Example'
        };
    },

    async created() {
        this.get_version();
        this.resetPars();
    },

    filters: {
        to2sf(value) {
            return Number(value).toFixed(2);
        }
    },

    computed: {
        isRunDisabled: function () {
            console.log(this.paramError);
            return this.paramError && Object.keys(this.paramError).length > 0;
        }
    },

    methods: {

        async get_version() {
            const response = await sciris.rpc('get_version');
            this.version = response.data;
        },

        async runSim() {
            this.running = true;
            // this.graphs = this.$options.data().graphs; // Uncomment this to clear the graphs on each run
            this.err = this.$options.data().err;

            console.log(this.status);
            console.log(this.sim_pars, this.epi_pars);

            // Run a a single sim
            try {
                const response = await sciris.rpc('run_sim', [this.sim_pars, this.epi_pars, this.show_animation]);
                this.result.graphs = response.data.graphs;
                this.result.files = response.data.files;
                this.result.summary = response.data.summary;
                this.err = response.data.err;
                this.sim_pars = response.data.sim_pars;
                this.epi_pars = response.data.epi_pars;
                this.history.push(JSON.parse(JSON.stringify({ sim_pars: this.sim_pars, epi_pars: this.epi_pars, result: this.result })));
                this.historyIdx = this.history.length - 1;

            } catch (e) {
                this.err = 'Error running model: ' + e;
            }
            this.running = false;

        },

        async resetPars() {
            const response = await sciris.rpc('get_defaults', [this.reset_choice]);
            this.sim_pars = response.data.sim_pars;
            this.epi_pars = response.data.epi_pars;
            this.setupFormWatcher('sim_pars');
            this.setupFormWatcher('epi_pars');
            this.graphs = [];
        },
        setupFormWatcher(paramKey) {
            const params = this[paramKey];
            if (!params) {
                return;
            }
            Object.keys(params).forEach(key => {
                this.$watch(`${paramKey}.${key}`, this.validateParam(key), { deep: true });
            });
        },
        validateParam(key) {
            return (param) => {
                if (param.best <= param.max && param.best >= param.min) {
                    this.$delete(this.paramError, key);
                } else {
                    this.$set(this.paramError, key, `Please enter a number between ${param.min} and ${param.max}`);
                }
            };
        },

        async downloadPars() {
            const d = new Date();
            const datestamp = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}_${d.getHours()}.${d.getMinutes()}.${d.getSeconds()}`;
            const fileName = `COVASim_parameters_${datestamp}.json`;

            // Adapted from https://stackoverflow.com/a/45594892 by Gautham
            const data = {
                sim_pars: this.sim_pars,
                epi_pars: this.epi_pars,
            };
            const fileToSave = new Blob([JSON.stringify(data, null, 4)], {
                type: 'application/json',
                name: fileName
            });
            saveAs(fileToSave, fileName);
        },

        async uploadPars() {
            try {
                const response = await sciris.upload('upload_pars');  //, [], {}, '');
                this.sim_pars = response.data.sim_pars;
                this.epi_pars = response.data.epi_pars;
                this.graphs = [];
            } catch (error) {
                sciris.fail(this, 'Could not upload parameters', error);
            }
        },

        loadPars() {
            this.sim_pars = this.history[this.historyIdx].sim_pars;
            this.epi_pars = this.history[this.historyIdx].epi_pars;
            this.result = this.history[this.historyIdx].result;
        },

        async downloadExcel() {
            const res = await fetch(this.result.files.xlsx.content);
            const blob = await res.blob();
            saveAs(blob, this.result.files.xlsx.filename);
        },

        async downloadJson() {
            const res = await fetch(this.result.files.json.content);
            const blob = await res.blob();
            saveAs(blob, this.result.files.json.filename);
        },

    },

});

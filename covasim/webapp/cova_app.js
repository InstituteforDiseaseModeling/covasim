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
            if (this.graph['json']){
                let x = JSON.parse(this.graph.json);
                x.responsive = true;
                Plotly.react(this.graph.id, x);
                window.dispatchEvent(new Event('resize'))
            }
        });
    },
    updated() {
        this.$nextTick(function () {
            if (this.graph['json']){
                let x = JSON.parse(this.graph.json);
                x.responsive = true;
                Plotly.react(this.graph.id, x);
            } else {
                Plotly.purge(this.graph.id)
            }
        });
    }
};

const interventionTableConfig = {
    social_distance: {
        formTitle: "Distancing",
        toolTip: "Physical distancing and social distancing interventions",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', label: 'Effectiveness', tooltip: 'Impact of social distancing (examples: 20 = mild, 50 = moderate, 80 = aggressive)', min: 0, max: 100, value: 50}
        ],
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            return {start, end, level};
        }
    },
    school_closures: {
        formTitle: "Schools",
        toolTip: "School and university closures",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', label: 'Effectiveness', tooltip: 'Impact of school closures (0 = no schools closed, 100 = all schools closed)', min: 0, max: 100, value: 90}
        ],
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            return {start, end, level};
        }
    },
    symptomatic_testing: {
        formTitle: "Testing",
        toolTip: "Testing rates for people with symptoms",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', label: 'Effectiveness', tooltip: 'Proportion of people tested per day (0 = no testing, 10 = 10% of people tested per day, 100 = everyone tested every day); assumes 1 day test delay', min: 0, max: 100, value: 10}
        ],
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            return {start, end, level};
        }
    },
    contact_tracing: {
        formTitle: "Tracing",
        toolTip: "Contact tracing of diagnosed cases (requires testing intervention)",
        fields: [
            {key: 'start', type: 'number', label: 'Start day', tooltip: 'Start day of intervention', value: 0},
            {key: 'end', type: 'number', label: 'End day', tooltip: 'End day of intervention (leave blank for no end)', value: null},
            {key: 'level', type: 'number', label: 'Effectiveness', tooltip: 'Effectiveness of contact tracing (0 = no tracing, 100 = all contacts traced); assumes 1 day tracing delay. Please note: you must implement a testing intervention as well for tracing to have any effect.', min: 0, max: 100, value: 80}
        ],
        handleSubmit: function(event) {
            const start = vm.parse_day(event.target.elements.start.value);
            const end = vm.parse_day(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            return {start, end, level};
        }
    }

};

function copyright_year() {
    const release_year = 1999
    const current_year = new Date().getFullYear()
    let range = [release_year]

    if (current_year > release_year){
        range.push(current_year)
    }

    return range.join("-")
}

function generate_upload_file_handler(onsuccess, onerror) {
    return function(e){
            let files = e.target.files;
            if (files.length > 0){
                const data = new FormData();
                data.append('uploadfile', files[0])
                data.append('funcname', 'upload_file')
                data.append('args', undefined)
                data.append('kwargs', undefined)
                fetch("/api/rpcs", {
                  "body": data,
                  "method": "POST",
                  "mode": "cors",
                  "credentials": "include"
                }).then(response => {
                    if(!response.ok){
                        throw new Error(response.json())
                    }
                    return response.text()
                }).then(data => {
                    remote_filepath = data.trim()
                                          .replace(/["]/g, "")
                    onsuccess(remote_filepath)
                })
                .catch(error => {
                    if (onerror){
                        sciris.fail(this, "Could not upload file.", error)
                    } else {
                        onerror(error)
                    }
                })
            } else {
                console.warn("No input file selected.")
            }
        }
}

var vm = new Vue({
    el: '#app',

    components: {
        'plotly-chart': PlotlyChart,
    },

    data() {
        return {
            debug: false,
            app: {
                title: "Covasim",
                version: 'Unable to connect to server!', // This text will display instead of the version
                copyright_year: copyright_year(),
                copyright_owner: "Bill & Melinda Gates Foundation",
                github_url: "https://github.com/institutefordiseasemodeling/covasim",
                org_url: "https://idmod.org",
                docs_url: "http://docs.covasim.org",
                paper_url: "http://paper.covasim.org",
                publisher_url: "https://gatesfoundation.org",
                license: 'Loading...',
                notice: 'Loading...'
            },
            panel_open: true,
            panel_width: null,
            resizing: false,
            history: [],
            historyIdx: 0,
            sim_length: {},
            sim_pars: {},
            epi_pars: {},
            datafile: {
                local_path: null,
                server_path: null
            },
            int_pars: {},
            intervention_figs: {},
            show_animation: false,
            result: { // Store currently displayed results
                graphs: [],
                summary: {},
                files: {},
            },
            paramError: {},
            scenarioError: {},
            interventionTableConfig,
            running: false,
            errs: [],
            reset_options: ['Default', 'Optimistic', 'Pessimistic'],
            reset_choice: 'Default'
        };
    },

    created() {
        this.get_version();
        this.get_location_options();
        this.resetPars();
        this.watchSimLengthParam();
        this.get_licenses();
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
        },
        is_debug: function () {
            return this.debug || /debug=true/i.test(window.location.search)
        }
    },

    methods: {
        async addIntervention(scenarioKey, event) {
            const intervention = this.interventionTableConfig[scenarioKey].handleSubmit(event);
            const key = scenarioKey;
            const self = this
            if (!this.int_pars[key]) {
                this.$set(this.int_pars, key, []);
            }
            // validate intervention
            const notValid = !intervention.end || intervention.start < 0 || intervention.end <= intervention.start
            if (notValid) {
                this.$set(this.scenarioError, scenarioKey, `Please enter a valid day range`);
                return;
            }

            const overlaps = this.int_pars[key].some(({start, end}) => {
                return start <= intervention.start && end >= intervention.start ||
                    start <= intervention.end && end >= intervention.end ||
                    intervention.start <= start && intervention.end >= end;
            })
            if (overlaps){
                this.$set(this.scenarioError, scenarioKey, `Interventions of the same type cannot have overlapping day ranges.`)
                return ;
            }

            const outOfBounds = intervention.start > this.sim_length.best || intervention.end > this.sim_length.best || this.int_pars[key].some(({start, end}) => {
                return start > self.sim_length.best || end > self.sim_length.best
            })
            if (outOfBounds){
                this.$set(this.scenarioError, scenarioKey, `Intervention cannot start or end after the campaign duration.`)
                return;
            }
            this.$set(this.scenarioError, scenarioKey, '');

            this.int_pars[key].push(intervention);
            const result = this.int_pars[key].sort((a, b) => a.start - b.start);
            this.$set(this.int_pars, key, result);
            const response = await sciris.rpc('get_gantt', undefined, {int_pars: this.int_pars, intervention_config: this.interventionTableConfig, n_days: this.sim_length.best});
            this.intervention_figs = response.data;
        },
        async deleteIntervention(scenarioKey, index) {
            this.$delete(this.int_pars[scenarioKey], index);
            const response = await sciris.rpc('get_gantt', undefined, {int_pars: this.int_pars, intervention_config: this.interventionTableConfig});
            this.intervention_figs = response.data;
        },

        parse_day(day) {
            if (day == null || day == '') {
                const output = this.sim_length.best
                return output
            } else {
                const output = parseInt(day)
                return output
            }
        },

        resize_start() {
            this.resizing = true;
        },
        resize_end() {
            this.resizing = false;
        },
        resize_apply(e) {
            if (this.resizing) {
                // Prevent highlighting
                e.stopPropagation();
                e.preventDefault();
                this.panel_width = (e.clientX / window.innerWidth) * 100;
            }
        },

        dispatch_resize(){
            window.dispatchEvent(new Event('resize'))
        },
        async get_version() {
            const response = await sciris.rpc('get_version');
            this.app.version = response.data;
        },

        async get_location_options() {
            let response = await sciris.rpc('get_location_options');
            for (let country of response.data) {
                this.reset_options.push(country);
            }
        },

        async get_licenses() {
            const response = await sciris.rpc('get_licenses');
            this.app.license = response.data.license;
            this.app.notice = response.data.notice;
        },

        async runSim() {
            this.running = true;
            // this.graphs = this.$options.data().graphs; // Uncomment this to clear the graphs on each run
            this.errs = this.$options.data().errs;

            console.log('status:', this.status);

            // Run a a single sim
            try {
                if(this.datafile.local_path === null){
                    this.reset_datafile()
                }
                const kwargs = {
                    sim_pars: this.sim_pars,
                    epi_pars: this.epi_pars,
                    int_pars: this.int_pars,
                    datafile: this.datafile.server_path,
                    show_animation: this.show_animation,
                    n_days: this.sim_length.best,
                    location: this.reset_choice
                }
                console.log('run_sim: ', kwargs);
                const response = await sciris.rpc('run_sim', undefined, kwargs);
                this.result.graphs = response.data.graphs;
                this.result.files = response.data.files;
                this.result.summary = response.data.summary;
                this.errs = response.data.errs;
                // this.panel_open = this.errs.length > 0; // Better solution would be to have a pin button
                this.sim_pars = response.data.sim_pars;
                this.epi_pars = response.data.epi_pars;
                this.int_pars = response.data.int_pars;
                this.history.push(JSON.parse(JSON.stringify({ sim_pars: this.sim_pars, epi_pars: this.epi_pars, reset_choice: this.reset_choice, int_pars: this.int_pars, result: this.result })));
                this.historyIdx = this.history.length - 1;

            } catch (e) {
                this.errs.push({
                    message: 'Unable to submit model.',
                    exception: `${e.constructor.name}: ${e.message}`
                })
                this.panel_open = true
            }
            this.running = false;

        },

        async resetPars() {
            const response = await sciris.rpc('get_defaults', [this.reset_choice]);
            this.sim_pars = response.data.sim_pars;
            this.epi_pars = response.data.epi_pars;
            this.sim_length = this.sim_pars['n_days'];
            this.int_pars = {};
            this.intervention_figs = {};
            this.setupFormWatcher('sim_pars');
            this.setupFormWatcher('epi_pars');
            // this.result.graphs = [];
            this.reset_datafile()
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

        watchSimLengthParam() {
            this.$watch('sim_length', this.validateParam('sim_length'), { deep: true });
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
            const fileName = `covasim_parameters_${datestamp}.json`;

            // Adapted from https://stackoverflow.com/a/45594892 by Gautham
            const data = {
                sim_pars: this.sim_pars,
                epi_pars: this.epi_pars,
                int_pars: this.int_pars
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
                this.int_pars = response.data.int_pars;
                this.result.graphs = [];
                this.intervention_figs = {}

                if (this.int_pars){
                    const gantt = await sciris.rpc('get_gantt', undefined, {int_pars: this.int_pars, intervention_config: this.interventionTableConfig});
                    this.intervention_figs = gantt.data;
                }

            } catch (error) {
                sciris.fail(this, 'Could not upload parameters', error);
            }
        },
        upload_datafile: generate_upload_file_handler(function(filepath){
            vm.datafile.server_path = filepath
        }),

        reset_datafile() {
            this.datafile = {
                local_path: null,
                server_path: null
            }
        },

        loadPars() {
            this.sim_pars = this.history[this.historyIdx].sim_pars;
            this.epi_pars = this.history[this.historyIdx].epi_pars;
            this.reset_choice = this.history[this.historyIdx].reset_choice;
            this.int_pars = this.history[this.historyIdx].int_pars;
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

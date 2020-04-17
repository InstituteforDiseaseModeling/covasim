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
            }
        }
        );
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
        formTitle: "Social Distancing",
        fields: [{key: 'start', type: 'number', label: 'Start Day'},
            {key: 'end', type: 'number', label: 'End Day'},
            {label: 'Effectiveness', key: 'level', type: 'select', options: [{label: 'Aggressive Effectiveness', value: 'aggressive'}, {label: 'Moderate Effectiveness', value: 'moderate'}, {label: 'Mild Effectiveness', value: 'mild'}]}],
        handleSubmit: function(event) {
            const start = parseInt(event.target.elements.start.value);
            const end = parseInt(event.target.elements.end.value);
            const level = event.target.elements.level.value;
            return {start, end, level};
        }
    },
    school_closures: {
        formTitle: "School Closures",
        fields: [{key: 'start', type: 'number', label: 'Start Day'}, {key: 'end', type: 'number', label: 'End Day'}],
        handleSubmit: function(event) {
            const start = parseInt(event.target.elements.start.value);
            const end = parseInt(event.target.elements.end.value);
            return {start, end};
        }
    },
    symptomatic_testing: {
        formTitle: "Symptomatic Testing",
        fields: [{key: 'start', type: 'number', label: 'Start Day'}, {key: 'end', type: 'number', label: 'End Day'}, {label: 'Accuracy', key: 'level', type: 'select', options: [{label: '60% Accuracy', value: '60'}, {label: '90% Accuracy', value: '90'},]}],
        handleSubmit: function(event) {
            const start = parseInt(event.target.elements.start.value);
            const end = parseInt(event.target.elements.end.value);
            const level = parseInt(event.target.elements.level.value);
            return {start, end, level};
        }
    },
    contact_tracing: {
        formTitle: "Contact Tracing",
        fields: [{key: 'start', type: 'number', label: 'Start Day'}, {key: 'end', type: 'number', label: 'End Day'}],
        handleSubmit: function(event) {
            const start = parseInt(event.target.elements.start.value);
            const end = parseInt(event.target.elements.end.value);
            return {start, end};
        }
  }

};

function copyright_year() {
    const release_year = 2020
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
                github_url: "https://github.com/institutefordiseasemodeling/covasim",
                org_url: "https://idmod.org",
                docs_url: "https://institutefordiseasemodeling.github.io/covasim-docs",
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
            intervention_pars: {},
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
            reset_options: ['Example'],//, 'Seattle', 'Wuhan', 'Global'],
            reset_choice: 'Example'
        };
    },

    async created() {
        this.get_version();
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
            if (!this.intervention_pars[key]) {
                this.$set(this.intervention_pars, key, []);
            }
            // validate intervention
            const notValid = !intervention.end || !intervention.start || intervention.end <= intervention.start
            if (notValid) {
                this.$set(this.scenarioError, scenarioKey, `Please enter a valid day range`);
                return;
            }

            const overlaps = this.intervention_pars[key].some(({start, end}) => {
                return start <= intervention.start && end >= intervention.start ||
                    start <= intervention.end && end >= intervention.end ||
                    intervention.start <= start && intervention.end >= end;
            })
            if (overlaps){
                this.$set(this.scenarioError, scenarioKey, `Interventions of the same type cannot have overlapping day ranges.`)
                return ;
            }

            const outOfBounds = intervention.start > this.sim_length.best || intervention.end > this.sim_length.best || this.intervention_pars[key].some(({start, end}) => {
                return start > self.sim_length.best || end > self.sim_length.best
            })
            if (outOfBounds){
                this.$set(this.scenarioError, scenarioKey, `Intervention cannot start or end after the campaign duration.`)
                return;
            }
            this.$set(this.scenarioError, scenarioKey, '');

            this.intervention_pars[key].push(intervention);
            const result = this.intervention_pars[key].sort((a, b) => a.start - b.start);
            this.$set(this.intervention_pars, key, result);
            const response = await sciris.rpc('get_gantt', undefined, {intervention_pars: this.intervention_pars, intervention_config: this.interventionTableConfig});
            this.intervention_figs = response.data;
        },
        async deleteIntervention(scenarioKey, index) {
            this.$delete(this.intervention_pars[scenarioKey], index);
            const response = await sciris.rpc('get_gantt', undefined, {intervention_pars: this.intervention_pars, intervention_config: this.interventionTableConfig});
            this.intervention_figs = response.data;
        },
        open_panel() {
            this.panel_open = true;
        },
        close_panel() {
            this.panel_open = false;
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

        async get_version() {
            const response = await sciris.rpc('get_version');
            this.app.version = response.data;
        },
        async get_licenses(){
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
                    intervention_pars: this.intervention_pars,
                    datafile: this.datafile.server_path,
                    show_animation: this.show_animation,
                    n_days: this.sim_length.best
                }
                console.log('run_sim: ', kwargs);
                const response = await sciris.rpc('run_sim', undefined, kwargs);
                this.result.graphs = response.data.graphs;
                this.result.files = response.data.files;
                this.result.summary = response.data.summary;
                this.errs = response.data.errs;
                this.panel_open = this.errs.length > 0;
                this.sim_pars = response.data.sim_pars;
                this.epi_pars = response.data.epi_pars;
                this.intervention_pars = response.data.intervention_pars;
                this.history.push(JSON.parse(JSON.stringify({ sim_pars: this.sim_pars, epi_pars: this.epi_pars, intervention_pars: this.intervention_pars, result: this.result })));
                this.historyIdx = this.history.length - 1;

            } catch (e) {
                this.errs.push({
                    message: 'Unable to submit model.',
                    exception: `${e.constructor.name}: ${e.message}`
                })
            }
            this.running = false;

        },

        async resetPars() {
            const response = await sciris.rpc('get_defaults', [this.reset_choice]);
            this.sim_pars = response.data.sim_pars;
            this.epi_pars = response.data.epi_pars;
            this.sim_length = {...this.sim_pars['n_days']}
            this.intervention_pars = {};
            this.intervention_figs = {};
            this.setupFormWatcher('sim_pars');
            this.setupFormWatcher('epi_pars');
            this.graphs = [];
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
                intervention_pars: this.intervention_pars
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
                this.intervention_pars = response.data.intervention_pars;
                this.graphs = [];
                this.intervention_figs = {}

                if (this.intervention_pars){
                    const gantt = await sciris.rpc('get_gantt', undefined, {intervention_pars: this.intervention_pars, intervention_config: this.interventionTableConfig});
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
            this.intervention_pars = this.history[this.historyIdx].intervention_pars;
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

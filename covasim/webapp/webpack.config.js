const path = require('path');
const webpack = require('webpack');
const { VueLoaderPlugin } = require('vue-loader')
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = {
    entry: { app: './cova_app.js' },
    resolve: {
        alias: {
            vue$: "vue/dist/vue.esm.js",
            // Alias for using source of BootstrapVue
            'bootstrap-vue$': 'bootstrap-vue/src/index.js'
        }
    },
    module: {
        rules: [
            {
                test: /\.vue$/,
                loader: 'vue-loader'
            },
            {
                test: /\.css$/,
                use: [
                    'vue-style-loader',
                    MiniCssExtractPlugin.loader, // instead of style-loader
                    'css-loader'
                ]
            }
        ]
    },
    optimization: {
        chunkIds: "named",
        splitChunks: {
            cacheGroups: {
                common: {
                    test: /node_modules/,
                    chunks: "initial",
                    name: "vendor",
                    priority: 10,
                    enforce: true
                }
            }
        }
    },
    plugins: [
        new CleanWebpackPlugin(),
        new webpack.SourceMapDevToolPlugin({
            filename: '[file].map',
            exclude: ['vendor.js', /\.css/]
        }),
        new VueLoaderPlugin(),
        new MiniCssExtractPlugin()
    ],

    output: {
        filename: '[name].js',
        path: path.resolve(__dirname, 'dist'),
    },
    externals: {
        "plotly": "Plotly"
    }
};

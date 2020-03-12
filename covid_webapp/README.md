# Run web UI

The web UI consists of

- Static files e.g. `index.html`
- A Flask app in `covid_app.py` wrapped in a Sciris App

It can be served in two ways

## Quick local testing

To run the app locally via Twisted, simply run

```shell script
./launch_webapp
```

The site will be accessible at `http://localhost:8188` or whichever port is specified in `launch_webapp`

## Deployment

Recommended deployment is using `nginx` to serve the static files, and `gunicorn` to run the Flask app.

### Requirements

You must have nginx (`sudo apt install nginx`) and gunicorn (`pip install gunicorn`) installed. 

### Set up nginx

1. Copy `example_nginx_config` to e.g. `/etc/nginx/sites-enabled/covid` (can change filename if desired)
2. Edit the copied file to specify
    - The hostname/URL for the site e.g. `voi.idmod.org`
    - The full path to the directory containing `index.html` on the system running `nginx`
    - Change the port in `proxy_pass` line if desired - it must match the port in `launch_gunicorn`
3. Reload or restart `nginx` e.g. `sudo service nginx reload`

For example, this will start it running at `localhost:8188`:

```script
server {
    listen 8188;
    server_name localhost;
    location / {
        root /home/my_username/covid_abm/covid_webapp;
    }
    location /api {
        proxy_pass http://127.0.0.1:8097/;
    }
}
```

### Run gunicorn

1. Edit `launch_gunicorn` to set the number of workers as desired - usual recommendation is twice the number of CPUs but it might be better for this app to just run the number of CPUs because the RPCs are computationally expensive
2. Run `launch_gunicorn`. This will need to be kept running to support the site (so run via `nohup` or `screen` etc.).

For example:
```script
cd covid_abm/covid_webapp
screen -S covid_app
./launch_gunicorn
<Ctrl+D>

...

screen -R covid_app
```

Note that for local development, you can add the `--reload` flag to the `gunicorn` command to automatically reload the site. This can be helpful if using the `nginx+gunicorn` setup for local development.
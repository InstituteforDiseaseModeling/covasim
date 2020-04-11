Covasim Platform
================

This contains files to run Covasim on Kubernetes. This is meant for testing Kubernetes Locally at the moment. 

# Running Locally

## Setup environment

On Linux machines that support snaps, you can use the following steps to obtain kubernetes for testing

First install the microk8s snap

```bash
sudo snap install microk8s --classic
```

Verify it is running

```bash
sudo microk8s status
```

If it is not running, you can start it using
```bash
sudo microk8s start
```

Optionally you can alias the snap's kubectl script
```bash
sudo snap alias microk8s.kubectl kubectl
```


You will also need to do the following to setup permission to run the kubectl commands

```bash
sudo usermod -a -G microk8s $(whoami)
sudo chown -f -R $(whoami) ~/.kube
```

After running those commands you will to logout and login back in to register new groups. You can also perform `su - $(whoami)` to get a new shell with the group changes loaded.

Lastly, enable DNS addon
```bash
sudo microk8s.enable dashboard dns
sudo microk8s.enable ingress
``` 

## Deploy 
Deploy the app using
```bash
kubectl apply -f covasim-deployment.yaml
kubectl apply -f ingress-local.yaml
```

When deploying to azure, you want to use the `ingress-azure.yaml` ingress file.

Check the status of the app using
```bash
kubectl get pods
kubectl get service
```

The service should now available at http://localhost:8000

To stop the application run
```bash
kubectl delete deployment.apps/covasim service/covasim
```


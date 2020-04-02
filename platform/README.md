Covasim Platform
================

This contains files to run Covasim on Kubernetes. 



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

The steps under [Running](#Running) assumes you have

You will also need to do the following to setup permission to run the kubectl commands

```bash
sudo usermod -a -G microk8s $(whoami)
sudo chown -f -R $(whoami) ~/.kube
```

After running those commands you will to logout and login back in to register new groups. You can also perform `su - $(whoami)` to get a new shell with the group changes loaded.

Lastly, enable DNS addon
```bash
sudo microk8s.enable dashboard dns
``` 

## Deploy 
Deploy the app using
```bash
kubectl apply -f covasim-service.yaml

```

Check the status of the app using
```bash
kubectl get pods
kubectl get service
```

Find the internal ip using `echo "http://$(kubectl get all --all-namespaces | grep service/covasim | awk '{print $4}')"`

To stop the application run
```bash
kubectl delete service/covasim pod/covasim
```


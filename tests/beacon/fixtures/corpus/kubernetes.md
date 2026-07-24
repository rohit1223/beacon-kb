# Kubernetes Container Orchestration

Kubernetes, often abbreviated as k8s, is an open-source system for automating deployment, scaling, and management of containerized applications.
The basic unit of scheduling in Kubernetes is the Pod.
A Pod encapsulates one or more containers that share network namespace and storage volumes.

Deployments manage the desired state for stateless applications.
A Deployment declares how many replicas of a Pod should be running and rolls out updates gradually using rolling update strategy.
When you update the container image in a Deployment, Kubernetes performs a rolling restart, replacing old pods with new ones without downtime.

Services expose a set of Pods as a stable network endpoint.
A ClusterIP Service is reachable only within the cluster.
A NodePort Service opens a port on every node.
A LoadBalancer Service provisions an external load balancer from the cloud provider.

ConfigMaps and Secrets inject configuration data and sensitive values into Pods as environment variables or mounted files.
Kubernetes namespaces partition cluster resources between teams.
Resource requests and limits control how much CPU and memory each container can consume.

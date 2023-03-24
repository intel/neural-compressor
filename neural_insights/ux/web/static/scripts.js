fetch('https://10.213.22.182:5000/api/workloads?token=asd')
    .then((response) => {
        return response.text();
    })
    .then((data) => {
        const workloadsData = JSON.parse(data);
        console.log(workloadsData)
        for (const workload of workloadsData.workloads) {
            var creationTime = new Date(workload.creation_time*1000);
            let workloadString = "<b>UUID: </b>" + workload.uuid + "<br><b>Framework:</b> " + workload.framework + "<br><b>Mode:</b> " + workload.mode + "<br><b>Model path:</b> " + workload.model_path + "<br><b>Workload location: </b>" + workload.workload_location + "<br><b>Created: </b>" + creationTime + "<br><br>";
            let div = document.createElement("div");
            div.className = "workloadBlock";
            div.innerHTML = workloadString;
            document.body.appendChild(div);
        }
    })

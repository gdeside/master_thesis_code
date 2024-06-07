import datetime
import subprocess
from six import iteritems

def submit_job(job_info, sbatch_script_path):
    """
    Submit a job to the Slurm Workload Manager using a crafted sbatch.

    :param job_info: The parameters to use in the sbatch.
    :param sbatch_script_path: The path to save the generated sbatch script.
    :return job_id: The id of the submitted job.
    """
    # Construct sbatch script content
    script_content = "#!/bin/bash\n"

    for key, value in iteritems(job_info):
        # Check for special case keys
        if key == "cpus_per_task":
            key = "cpus-per-task"
        if key == "mem_per_cpu":
            key = "mem-per-cpu"
        if key == "mail_user":
            key = "mail-user"
        if key == "mail_type":
            key = "mail-type"
        if key == "job_name":
            key = "job-name"
        elif key == "script":
            continue
        script_content += f"#SBATCH --{key}={value}\n"

    script_content += f"\n{job_info['script']}"

    # Save the script content to the specified file
    with open(sbatch_script_path, 'w') as script_file:
        script_file.write(script_content)

    print("[INFO] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") +
          f": Generated sbatch script: '{sbatch_script_path}'")

    # Run sbatch command as subprocess.
    try:
        sbatch_output = subprocess.check_output(["sbatch", sbatch_script_path])
    except subprocess.CalledProcessError as e:
        # Print error message from sbatch for easier debugging, then pass on exception
        if sbatch_output is not None:
            print("ERROR: Subprocess call output: %s" % sbatch_output)
        raise e

    # Parse job id from sbatch output.
    sbatch_output = sbatch_output.decode().strip("\n ").split()
    for s in sbatch_output:
        if s.isdigit():
            job_id = int(s)
            return job_id

# Example usage
job_info = {
    "job_name": "MF_submission_mgh_1005_pair",
    "time": "28:00:00",
    "ntasks": 1,
    "cpus_per_task": 25,
    "mem_per_cpu": 10000,
    "mail_user": "guillaume.deside@student.uclouvain.be",
    "mail_type": "ALL",
    "output": "output_MF_mgh_1005_pair.txt",
    "script": "python3 patient_mf_analysis.py"
}

sbatch_script_path = "submit_MF_mgh_1005_pair.sbatch"

submit_job(job_info, sbatch_script_path)



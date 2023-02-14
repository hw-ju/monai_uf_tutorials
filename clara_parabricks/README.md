# **Use Clara Parabricks on HiperGator**

This directory is a tutorial on how to use tools provided by Clara Parabricks on HiperGator, adapted from [the tutorial in Clara Parabricks v4.0.1 documentation](https://docs.nvidia.com/clara/parabricks/4.0.1/Tutorials.html).  

## **Note**
1. This tutorial assumes you have downloaded the repository `monai_uf_tutorials` following [this section](../README.md/#download-this-repository-on-hipergator).
2. In all following commands and scripts, replace `hju` with your HiperGator username; change the path to files according to your own settings on HiperGator. 
3. In all following SLURM job scripts, alter the `#SBATCH` settings for your own needs. 

    **IMPORTANT:** Refer to [the doc](https://docs.nvidia.com/clara/parabricks/4.0.1/GettingStarted.html) on how to allocate adequate computing resources (e.g., number of gpu, number of cpu core, amounts of memory) for using Clara Parabricks.
4. Please read the comments in each script to get a better understanding on how to tune the scripts to your own needs.


## **How to run**
0. Read [the tutorial in Clara Parabricks v4.0.1 documentation](https://docs.nvidia.com/clara/parabricks/4.0.1/Tutorials.html) to understand what will be done in this tutorial.

1. Go to directory `/monai_uf_tutorials/clara_parabricks/`
    ```
    cd ~/monai_uf_tutorials/clara_parabricks/
    ```

2. Build a Clara Parabricks container by submitting a SLURM job script [`build_container.sh`](./build_container.sh)
    ```
    sbatch build_container.sh
    ```
    Check if the job is done 
    ```
    squeue -u $USER
    ```
    You can check it in multiple ways, refer to [UFRC doc](https://help.rc.ufl.edu/doc/UFRC_Help_and_Documentation) for more details.
    
    When the job is done, check the SLURM job output 
    ```
    cat build_container.sh.job_id.output
    ```
    If the container was built successfully, there should be no error in the output file and the output file should look similar to [`build_container.sh.job_id.out`](./build_container.sh.job_id.out) .

4. Download sample data by [`download_sample_data.sh`](./download_sample_data.sh)    
    ```
    sbatch download_sample_data.sh
    ```
    If the data was downloaded successfully, there should be no error in the SLURM job output file and the output file should look similar to [`download_sample_data.sh.job_id.out`](./download_sample_data.sh.job_id.out).

5. Run alignment tool - FQ2BAM (FASTA + FASTQ ==> BAM) by [`fq2bam.sh`](./fq2bam.sh)
    ```
    sbatch fq2bam.sh
    ```
    Three output files should be generated (refer to [fq2bam tutorial](https://docs.nvidia.com/clara/parabricks/4.0.1/Tutorials/FQ2BAM_Tutorial.html) and [fq2bam doc](https://docs.nvidia.com/clara/parabricks/4.0.1/Documentation/ToolDocs/man_fq2bam.html#man-fq2bam)). 
    
    The SLURM job output file should be similar to [`fq2bam.sh.job_id.out`](./fq2bam.sh.job_id.out). 


6. Run the gold-standard GATK variant caller - HaplotypeCaller (BAM ==> VCF)  by [`haplotype.sh`](./haplotype.sh)
    ```
    sbatch haplotype.sh
    ```
    A `.vcf` file should be generated (refer to [HaplotypeCaller tutorial](https://docs.nvidia.com/clara/parabricks/4.0.1/Tutorials/HaplotypeCaller_Tutorial.html) and [HaplotypeCaller doc](https://docs.nvidia.com/clara/parabricks/4.0.1/Documentation/ToolDocs/man_haplotypecaller.html#man-haplotypecaller)). 
    
    The SLURM job output file should be similar to [`haplotype.sh.job_id.out`](./haplotype.sh.job_id.out).     

7. Run the deep-learning-based germline variant caller - DeepVariant (BAM ==> VCF)  by [`deepvariant.sh`](./deepvariant.sh)
    ```
    sbatch deepvariant.sh
    ```
    A `.vcf` file should be generated (refer to [DeepVariant doc](https://docs.nvidia.com/clara/parabricks/4.0.1/Documentation/ToolDocs/man_deepvariant.html#man-deepvariant)). 
    
    The SLURM job output file should be similar to [`deepvariant.sh.job_id.out`](./deepvariant.sh.job_id.out).      

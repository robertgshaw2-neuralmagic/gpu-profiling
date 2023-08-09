FROM ghcr.io/huggingface/text-generation-inference

RUN conda install jupyter
RUN conda install python-language-server

ENTRYPOINT ["/bin/bash"]

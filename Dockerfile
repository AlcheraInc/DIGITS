# Reference
#   https://gitlab.com/nvidia/digits/blob/master/6.0/Dockerfile

FROM nvidia/digits:latest

# remove packages installed by nvidia
RUN     pip uninstall -y digits
RUN     pip install --upgrade pip

VOLUME [ "/mnt/data", "/mnt/jobs" ,"/app/code" ]
WORKDIR /app

ENV DIGITS_JOBS_DIR=/mnt/jobs \
    DIGITS_LOGFILE_FILENAME=/mnt/jobs/digits.log

# reset
ENTRYPOINT []
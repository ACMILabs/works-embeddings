FROM python:3.10

RUN apt update && apt install make

COPY ./requirements/base.txt /code/requirements/base.txt
RUN pip install -Ur /code/requirements/base.txt

COPY . /code/
WORKDIR /code/

CMD ["bash", "./scripts/entrypoint.sh"]

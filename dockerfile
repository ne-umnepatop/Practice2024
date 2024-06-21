FROM python:3.12

RUN function install_deps() { \
    if command -v pacman &> /dev/null; then \
        pacman -S --noconfirm \
            wget \
            curl \
            pyenv \
    elif command -v apt-get &> /dev/null; then \
        apt-get update && apt-get install -y \
            build-essential \
            wget \
            curl \
            pyenv \
    else \
        echo "Not apt or pacman"; \
        exit 1; \
    fi; \
}
RUN install_deps
# RUN curl https://pyenv.run | bash
ENV PATH="/root/.pyenv/bin:$PATH"
RUN pyenv install 3.12
RUN pyenv global 3.12
RUN python -m venv /app/env
ENV PATH="/app/env/bin:$PATH"
RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    scipy \
    wheel \
    setuptools \
    gymnasium \
    'gymnasium[mujoco]' 

COPY . /app
WORKDIR /app
# RUN python -m unittest discover -s . -p "*_test.py"
CMD ["python", "app.py"]

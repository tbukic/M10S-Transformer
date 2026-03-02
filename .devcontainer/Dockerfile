ARG CUDA_VERSION="12.6.3-cudnn"
ARG CUDA_VARIANT="ubuntu22.04"
ARG UV_VERSION="0.9.21"
ARG USER="vscode"
ARG USER_UID="1000"
ARG USER_GID=${USER_UID}
ARG PYTHON_VERSION="3.13"
ARG NVM_VERSION="v0.40.3"
ARG NODE_VERSION="24.12.0"
ARG PROJDIR

ARG INCLUDE_NODE="y"
ARG INCLUDE_POSTGRESQL_CLIENT="y"
ARG INCLUDE_REDIS_TOOLS="n"
ARG INCLUDE_AWS_CLI="n"
ARG INCLUDE_AWS_SAM="n"
ARG INCLUDE_ANDROID_ADB="n"

ARG INCLUDE_PLAYWRIGHT_SYSTEM_DEPS="n"
ARG INCLUDE_OCR_PDF_STACK="n"
ARG INCLUDE_SAGEMATH="n"
ARG INCLUDE_GITHUB_CLI="n"
ARG INCLUDE_JEKYLL_LATEX="y"

ARG INCLUDE_EXPO_TOOLCHAIN="n"
ARG INCLUDE_VERCEL_CLI="n"
ARG INCLUDE_SUPABASE_CLI="n"
ARG INCLUDE_CDK_TOOLCHAIN="n"

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_image

FROM nvidia/cuda:${CUDA_VERSION}-devel-${CUDA_VARIANT} AS build-env

COPY --from=uv_image /uv /uvx /bin/

ARG USER
ARG USER_UID
ARG USER_GID
ARG PYTHON_VERSION
ARG NVM_VERSION
ARG NODE_VERSION
ARG PROJDIR

ARG INCLUDE_NODE
ARG INCLUDE_POSTGRESQL_CLIENT
ARG INCLUDE_REDIS_TOOLS
ARG INCLUDE_AWS_CLI
ARG INCLUDE_AWS_SAM
ARG INCLUDE_ANDROID_ADB

ARG INCLUDE_PLAYWRIGHT_SYSTEM_DEPS
ARG INCLUDE_OCR_PDF_STACK
ARG INCLUDE_SAGEMATH
ARG INCLUDE_GITHUB_CLI
ARG INCLUDE_JEKYLL_LATEX

ARG INCLUDE_EXPO_TOOLCHAIN
ARG INCLUDE_VERCEL_CLI
ARG INCLUDE_SUPABASE_CLI
ARG INCLUDE_CDK_TOOLCHAIN

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

SHELL ["/bin/bash", "-c"]

ENV UV_COMPILE_BYTECODE=0
ENV UV_LINK_MODE=copy

RUN set -eux; \
    PKGS="build-essential ca-certificates curl git gnupg2 jq less locales man-db openssh-client python3 python3-dev python3-pip python3-venv sudo tmux tree unzip vim wget zip"; \
    if [ "${INCLUDE_POSTGRESQL_CLIENT}" = "y" ]; then PKGS="${PKGS} postgresql-client"; fi; \
    if [ "${INCLUDE_REDIS_TOOLS}" = "y" ]; then PKGS="${PKGS} redis-tools"; fi; \
    if [ "${INCLUDE_ANDROID_ADB}" = "y" ]; then PKGS="${PKGS} android-tools-adb"; fi; \
    if [ "${INCLUDE_PLAYWRIGHT_SYSTEM_DEPS}" = "y" ]; then \
      PKGS="${PKGS} libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2t64"; \
    fi; \
    if [ "${INCLUDE_OCR_PDF_STACK}" = "y" ]; then \
      PKGS="${PKGS} tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd poppler-utils libmagic1 pandoc libreoffice libpng-dev libjpeg-dev libtiff-dev"; \
    fi; \
    if [ "${INCLUDE_SAGEMATH}" = "y" ]; then PKGS="${PKGS} sagemath"; fi; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends ${PKGS}; \
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen; \
    locale-gen; \
    if [ "${INCLUDE_GITHUB_CLI}" = "y" ]; then \
      curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /usr/share/keyrings/githubcli-archive-keyring.gpg; \
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list; \
      apt-get update -y; \
      apt-get install -y --no-install-recommends gh; \
    fi; \
    if ! getent group ${USER_GID}; then groupadd --gid ${USER_GID} ${USER}; fi; \
    if id -u ${USER_UID} >/dev/null 2>&1; then \
      existing_user=$(getent passwd ${USER_UID} | cut -d: -f1); \
      usermod -l ${USER} -d /home/${USER} -m ${existing_user}; \
    else \
      useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USER}; \
    fi; \
    echo ${USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER}; \
    chmod 0440 /etc/sudoers.d/${USER}; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get clean

RUN if [ "${INCLUDE_JEKYLL_LATEX}" = "y" ]; then \
      apt-get update -y; \
      apt-get install -y ruby-full zlib1g-dev texlive-latex-base texlive-latex-extra \
        texlive-fonts-recommended texlive-fonts-extra texlive-xetex texlive-science \
        texlive-bibtex-extra biber latexmk; \
      rm -rf /var/lib/apt/lists/*; \
    fi

USER ${USER}
ENV USER=${USER}
ENV HOME=/home/${USER}

RUN uv python install ${PYTHON_VERSION}

ENV NVM_DIR="${HOME}/.nvm"
RUN if [ "${INCLUDE_NODE}" = "y" ]; then \
      curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh | bash && \
      . "${NVM_DIR}/nvm.sh" && \
      nvm install ${NODE_VERSION} && \
      nvm alias default ${NODE_VERSION} && \
      nvm use default && \
      npm install -g npm@latest pnpm@latest && \
      echo 'export NVM_DIR="${HOME}/.nvm"' >> ~/.bashrc && \
      echo '[ -s "${NVM_DIR}/nvm.sh" ] && \\. "${NVM_DIR}/nvm.sh"' >> ~/.bashrc; \
    fi

RUN if [ "${INCLUDE_EXPO_TOOLCHAIN}" = "y" ] && [ "${INCLUDE_NODE}" = "y" ]; then \
      . "${NVM_DIR}/nvm.sh" && npm install -g expo-cli@latest @expo/ngrok@latest eas-cli@latest; \
    fi

RUN if [ "${INCLUDE_VERCEL_CLI}" = "y" ] && [ "${INCLUDE_NODE}" = "y" ]; then \
      . "${NVM_DIR}/nvm.sh" && npm install -g vercel@latest; \
    fi

RUN if [ "${INCLUDE_CDK_TOOLCHAIN}" = "y" ] && [ "${INCLUDE_NODE}" = "y" ]; then \
      . "${NVM_DIR}/nvm.sh" && npm install -g aws-cdk typescript ts-node esbuild; \
    fi

RUN if [ "${INCLUDE_SUPABASE_CLI}" = "y" ]; then \
      SUPABASE_URL="$(curl -fsSL https://api.github.com/repos/supabase/cli/releases/latest | jq -r '.assets[] | select(.name|test("linux_amd64\\.tar\\.gz$")) | .browser_download_url' | head -n1)" && \
      test -n "${SUPABASE_URL}" && \
      curl -fsSL "${SUPABASE_URL}" -o /tmp/supabase.tar.gz && \
      mkdir -p "${HOME}/.local/bin" && \
      tar -xzf /tmp/supabase.tar.gz -C "${HOME}/.local/bin" supabase && \
      chmod +x "${HOME}/.local/bin/supabase" && \
      rm -f /tmp/supabase.tar.gz; \
    fi

RUN if [ "${INCLUDE_AWS_CLI}" = "y" ]; then \
      cd /tmp && \
      curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
      unzip awscliv2.zip && \
      sudo ./aws/install && \
      rm -rf aws awscliv2.zip; \
    fi

RUN if [ "${INCLUDE_AWS_SAM}" = "y" ]; then \
      cd /tmp && \
      curl -L "https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip" -o "samcli.zip" && \
      unzip samcli.zip -d sam-installation && \
      sudo ./sam-installation/install && \
      rm -rf sam-installation samcli.zip; \
    fi

RUN if [ "${INCLUDE_JEKYLL_LATEX}" = "y" ]; then \
      echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc && \
      echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc && \
      mkdir -p "$HOME/gems" && \
      export GEM_HOME="$HOME/gems" && export PATH="$HOME/gems/bin:$PATH" && \
      gem install jekyll bundler; \
    fi

RUN if [ "${INCLUDE_SAGEMATH}" = "y" ]; then \
      sage -python -m pip install --upgrade ipykernel && \
      sage -python -m ipykernel install --user --name sage --display-name "SageMath $(sage --version)"; \
    fi

# Claude Code CLI is always installed
RUN curl -fsSL claude.ai/install.sh | bash

ENV PATH="${HOME}/.nvm/versions/node/v${NODE_VERSION}/bin:${HOME}/.local/bin:${PROJDIR}/.venv/bin:${PATH}"

WORKDIR ${PROJDIR}

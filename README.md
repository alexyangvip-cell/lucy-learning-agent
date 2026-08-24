# 青少年个性化学习 Agent

首次启动会在项目内创建 `.venv`，并默认通过[清华 TUNA PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)安装 `requirements.txt` 中的依赖，因此需要联网并请您耐心等待几分钟。

## 创建开发环境

macOS：

需要 macOS 12 或更高版本。

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Windows：

```powershell
py -V:3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
copy .env.example .env
```

上面的命令使用推荐版本 Python 3.14。
如果本机安装的是 Python 3.11、3.12 或 3.13，也可以将命令中的 `3.14` 替换为对应版本。

启动项目
```bash
streamlit run app.py
```

## Agent 个性化与安全记忆

`SOUL.md` 用普通 Markdown 描述 Agent 名称、语气、表达习惯和默认回答方式。

`OWNER.md` 使用 `schema_version: 1` YAML Frontmatter 保存受管资料，正文保留用户手写备注。
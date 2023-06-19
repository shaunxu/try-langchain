from dotenv import load_dotenv
from langchain.chains import LLMChain, ConversationChain, RetrievalQA
import os
from llms.chatlgm import ChatGLM
from langchain.prompts.prompt import PromptTemplate
import langchain

langchain.debug = True

load_dotenv()
llm_model_name_or_path = os.environ.get("LLM_MODEL_NAME_OR_PATH")
embedding_model_name_or_path = os.environ.get("EMBEDDING_MODEL_NAME_OR_PATH")
vectorstore_persist_directory = os.environ.get("VECTORSTORE_PERSIST_DIRECTORY")

llm = ChatGLM()
llm.load_model(llm_model_name_or_path)

template = """请根据如下信息回答问题。
{context}

问题是
{input}

注意，你的答案只能从上述信息中获取，不要自行杜撰。如果无法获得答案，请直接回答“我不知道”。
"""

prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=template
)


chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False
)

# context = """
# 工作职责： 
# 1、负责运维平台的建设，包括自动化部署、监控、日志、灾备、持续集成等； 
# 2、负责线上业务系统的审核、部署、发布、监控、维护和优化，保障产品可用性； 
# 职位要求： 
# 1、计算机或相关专业本科及以上学历； 
# 2、3年以上运维工作经验； 
# 3、精通 Linux 操作系统、TCP/IP 等常用协议； 
# 4、精通或深入了解一家公有云产品，AWS、阿里云、腾讯云、微软云、青云等； 
# 5、熟练掌握 Shell、Python、JavaScript 等一种或多种编程语言； 
# 6、熟悉 MongoDB、Redis、ElasticSearch、Nginx、Node.js 等日常管理和维护； 
# 7、熟悉 Jenkins、Zabbix、ELK 等优先； 
# 8、熟悉 Docker、Kubernetes 等优先； 
# 9、学习能力强，强烈的责任心，可以在压力下工作，及时处理突发状况。\n\n
# cloud-architecture.png 
# 基础设施基于 AWS 来构建 使用 Terraform、Packer 来管理基础设施，配置信息存放在 Git 仓储上 采用 Kubernetes 进行容器化部署应用 基础服务包括 MongoDB、Redis、Elasticsearch 应用层包括 AWS NLB、Nginx、Node.js 使用 Ansible 来对系统进行维护 CI/CD 采用 PingCode Pipe、Jenkins、Helm 来实现 日志系统：ElasticSearch、Kibana、Fluentd 监控系统：Zabbix、Prometheus、Grafana 调用链追踪：Skywalking 安全防护：Wazuh、Openstar\n\n
# 版本号 v1.0.0 修订人 孙敬云 修订时间 2022年06月16日 本文只说明如何使用配置中心，而不会说明配置中心服务本身的技术实现。 配置中心 image.png 「配置中心」是把各个应用程序中的配置、参数、开关，全部放到一个地方进行管理的工具。应用程序可以随时通过「配置中心」提供的接口获取到其所需的配置。 如果没有「配置中心」，我们当然也可以使用，例如环境变量（configmap），的方式来管理应用程序的配置。但是随着微服务、分布式的发展，这些传统的配置管理方式将很难适应。例如： 如何动态生效配置 如何统一多个环境的配置 如何实现灰度发布 如何对配置进行版本管理 区别于通过环境变量（configmap）的方式来管理配置，配置中心有这些优势： 集中管理各个环境中各个应用程序的配置信息 拥有配置的版本管理、发布管理、环境同步、权限管理等能力 配置信息实时生效，无需重启应用程序 image.png （配置中心截图） 配置中心SDK 使用配置中心时，离不开配置中心的SDK，下图为配置模块和配置中心SDK的关系： image.png 部分步骤说明： 2.引用模块 （同过去的逻辑） import config from \"./config\"; 3.获取模板 （同过去的逻辑，只是概念上有些变化） “模板”指的是原始的配置文件，而“获取模板”指的就是获取到根据环境合并的配置文件。     switch (env) {\n        case \"development\":\n            configTemplate = _.merge({}, configs.default, configs.development);\n            break;\n        case \"local\":\n            configTemplate = _.merge({}, configs.default, configs.local);\n            break;\n        case \"production\":\n            configTemplate = _.merge({}, configs.default, configs.production);\n            break;\n        case \"on-premises\":\n            configTemplate = _.merge({}, configs.default, configs.onPremises);\n            break;\n        case \"on-premises-dev\":\n            configTemplate = _.merge({}, configs.default, configs.onPremisesDev);\n            break;\n        case \"test\":\n            configTemplate = _.merge({}, configs.default, configs.test);\n            break;\n        default:\n            configTemplate = _.merge({}, configs.default, configs.development);\n    } 4.模板转译 在配置模板（原始配置文件）中，有一些被标记为需要替换的配置（new MixConfigItem()），它们需要通过配置中心SDK完成转译，例如： {\n    apiPrefix: \"\",\n    apiAdminPrefix: \"/api/open/admin\",\n    url: {\n        domain: new MixConfigItem(\"OPEN_API_DOMAIN\", \"https://open.pingcode.com\"),\n        login: new MixConfigItem(\"OAUTH2_LOGIN_URL\", \"https://pingcode.com/signin\")\n    },\n    ports: {\n        api: new MixConfigItem(\"OPEN_API_PORT\", 8080, (v: any) => parseInt(v) || 8080),\n        admin: new MixConfigItem(\"OPEN_ADMIN_PORT\", 8080, (v: any) => parseInt(v) || 8080),\n        oauth2: new MixConfigItem(\"OPEN_OAUTH2_PORT\", 8080, (v: any) => parseInt(v) || 8080),\n        rpc: new MixConfigItem(\"OPEN_RPC_PORT\", 8080, (v: any) => parseInt(v) || 8080)\n    },\n    s3Prefix: {\n        avatar: new MixConfigItem(\"OPEN_API_AVATAR_PREFIX\", \"https://s3.cn-north-1.amazonaws.com.cn/lcavatar/\")\n    },\n    isUpgrade: false,\n    frequencyLimit: new MixConfigItem(\"OPEN_API_FREQ_LIMIT_IN_ONE_MIN\", 200, parseInt)\n} 在正式转译之前，如果启用了配置中心，那么SDK会先同步拉取远程配置，然后按照远程配置>环境变量>默认值的顺序进行配置值替换；如果没有启用配置中心，那么按照环境变量>默认值的顺序进行配置值替换。 5.同步拉取 配置中心SDK会向配置中心同步拉取配置，在拉取配置失败时，程序不会终止，而是按照环境变量>默认值的顺序进行配置值替换。不过如果启用了强制远程配置模式，那么在拉取配置失败时，程序会报错。 6.export 配置模块导出的是配置的实例，而非配置的模板 7.开启定时、8.异步拉取配置、9.更新内存 如果启用了配置中心，并设置了更新频率，那么配置中心SDK会定时异步拉取配置，如果配置发生了变化，则会根据配置模板更新内存中的配置实例。 应用程序使用配置中心 第一步：更新eros@2.6.10+ 第二步：替换配置文件中的配置值。因为只用PingCode公有云会使用配置中心，本地不会、私有部署也不会。因此只需修改应用程序的production.ts配置文件即可。例如： // production.ts\n\n
# import { PRODUCTION_CONFIG } from \"@atinc/eros/config\";\nimport { FullApplicationType } from \"@atinc/eros/enums\";\nimport { _ } from \"@atinc/chaos\";\nimport { Is } from \"@atinc/chaos/constants\";\nimport { MixConfigItem } from \"@atinc/eros/enums\";\n\nimage.png 研发 开发任务 模块设计 系统架构 基础框架 决策/战略 预判/研究 职级年限 运维 基础运维 运维研发 性能调优 系统架构 产品/设计 模块级设计 子产品级设计 子产品设计/数据运营 产品整体/风格设计 W15/M8 (科学家) - - W14/M7 (科学家) - - W13/M6 (资深专家) 导师 参与 W12/M5 (资深专家) 导师 辅助 3 W11/M4 (资深专家) 导师 独立 辅助 3 W10/M3 (专家) 导师 独立 2 W9/M2 (专家) 导师 独立 辅助 W8/M1 (专家) 导师 独立 辅助 2 W7 (骨干) 导师 独立 辅助 1 W6 (骨干) 独立 辅助 1 W5 (骨干) 辅助 1 W4 (助理) W3 (助理) 研发： 开发任务：模块下的开发任务 模块：子产品包含的模块（看板模块，Job，Scrum，共享用例） 系统架构：子产品独立负责 基础架构：wt-chaos、小程序框架、MVC框架【组件库、wt-eros、wt-rd-core、业务组件库】 运维： 基础运维：服务更新、脚本执行、服务搭建(MongoDB、Redis) 运维研发：运维工具开发，如自动化备份数据库并验证等 性能调优：深入了解操作系统、数据库等知识，可以针对业务情况对系统进行调优 系统架构：架构线上运行环境，具备标准化、易维护、高可用、可扩展等能力
# """
context = """
工作职责： 
1、负责运维平台的建设，包括自动化部署、监控、日志、灾备、持续集成等； 
2、负责线上业务系统的审核、部署、发布、监控、维护和优化，保障产品可用性； 
职位要求： 
1、计算机或相关专业本科及以上学历； 
2、3年以上运维工作经验； 
3、精通 Linux 操作系统、TCP/IP 等常用协议； 
4、精通或深入了解一家公有云产品，AWS、阿里云、腾讯云、微软云、青云等； 
5、熟练掌握 Shell、Python、JavaScript 等一种或多种编程语言； 
6、熟悉 MongoDB、Redis、ElasticSearch、Nginx、Node.js 等日常管理和维护； 
7、熟悉 Jenkins、Zabbix、ELK 等优先； 
8、熟悉 Docker、Kubernetes 等优先； 
9、学习能力强，强烈的责任心，可以在压力下工作，及时处理突发状况。\n\n
"""
input = "运维基础设施"

output = chain.predict(context=context, input=input)
print(output)
# 北半球冬季积雪覆盖收缩与阈值边界北移：雪水当量趋势的同化敏感性

`output/北半球冬季积雪变化研究报告.tex` 为 BNU LaTeX 主文件，同目录提供编译后的 PDF。正文、图注和参考文献在 `output/sections/`，由 CSV 生成的 LaTeX 表格在 `output/tables/`。封面按提供的 BNU 课程论文模板排版，课程为地球复杂系统，指导老师为夏炎，学生为张贵（学号 202621250017）。封面字段在主文件的导言区设置。

报告附录保留补充统计表和阈值敏感性图，另提供公开代码仓库链接、精简 AI 提示词摘要及辅助使用说明。Git 仓库收录分析、绘图、质量检查和 LaTeX 源码，生成数据与图件由 `.gitignore` 排除。复现需要在项目上一级目录提供原始 `data.zip`，并按下述顺序运行脚本。

## 数据与研究范围

分析读取工作区提供的 `data.zip`。脚本默认从本项目的上一级目录读取原归档。可通过参数指定另一位置。预处理直接读取归档中的三个 NetCDF，原始文件保留在归档内。计算只用 SCF、SWE、T2m，以及归档内的时间、坐标、分析掩膜和面积。外部文献只用于产品与方法引用，并给出 ERA5 已知同化变更的时间。

固定分析域含 11,844 个有效格点，1980–2022 年的 43 个 DJF 冬季均完整。月度值按真实日数加权。备用场可通过单精度乘加精确复现，正式分析采用双精度。SWE 空间代表性缺少明确定义，储量在全格点平均和有雪面积条件平均两种假设下计算。阈值雪线是 SCF 定义的二维地理覆盖边界，主阈值为 0.50。

## 运行

用 Python 3.11，在本目录执行。绘图与审计需要安装 Nature Figure 技能，默认检测本机对应的技能目录。其他安装位置可将环境变量 `NATURE_FIGURE_SKILL` 指向包含 `scripts/` 的技能根目录。所用版本记录于 `logs/versions.json`。

```powershell
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = '1'
python code/00_preprocess.py ../data.zip
python code/03_global_metrics.py
python code/04_grid_trends_mk_sen.py
python code/05_region_latitude_analysis.py
python code/06_temperature_response.py
python code/07_threshold_snowline.py
python code/07b_assimilation_sensitivity.py
python code/08_swe_tail.py
python output/agent_mechanism/assimilation_checks.py
python code/08_domain_figure.py
python code/08_trend_figures.py
python code/08_spatial_figures.py
python code/08_snowline_figures.py
python code/08_synthesis_figures.py
python code/09_export_latex_tables.py
```

在 `output/` 目录内以 XeLaTeX 编译两遍以更新目录与引用。参考文献在源文件中通过 `thebibliography` 排版。

```powershell
Set-Location output
xelatex -interaction=nonstopmode -halt-on-error "北半球冬季积雪变化研究报告.tex"
xelatex -interaction=nonstopmode -halt-on-error "北半球冬季积雪变化研究报告.tex"
Set-Location ..
python code/10_quality_control.py
```

交付版实际使用 Tectonic 0.17.0 的 XeTeX 引擎与本地 TeX 缓存完成编译。对应命令为在 `output/` 内运行 `tectonic --only-cached --keep-logs "北半球冬季积雪变化研究报告.tex"`。无缓存时可使用装有 CTeX 和 Fandol 字体的 XeLaTeX 发行版。构建及检查记录位于 `logs/`。

## 输出索引

| 位置 | 内容 |
| --- | --- |
| `snow_project_output/研究任务Prompt.md` | 原始研究任务与交付要求 |
| `output/processed/` | 冬季字段与格点长期趋势 NetCDF |
| `output/analysis_config.yaml`、`output/region_masks/` | 时段、固定窗口与纬带、HAC 和区块重抽样设定 |
| `output/csv/` | 覆盖审计、备用核对、趋势、区域与纬带对比、去趋势响应、跨冬季滞后、阈值与断点敏感性 |
| `output/figures/` | 10 幅核心图、1 幅机制边界图与 1 幅阈值附图，PDF、SVG、600 dpi PNG |
| `output/agent_trend/`、`output/agent_spatial/`、`output/agent_climate/`、`output/agent_snowline/` | 模块说明及源数据 |
| `output/agent_mechanism/` | 科学复核与同化变更诊断 |
| `logs/` | 数据来源与元数据审计、运行日志、图件面板对齐、PDF 字号和碰撞检查 |

主分析使用固定域和原序列。2004 年水平项是已知产品时间一致性问题的敏感性检验。少数高 SWE 格点的固定剔除仅用于检验储量基线和趋势对高值的敏感度。主阈值边界分析分别保留逐年可用经度和 43 个冬季共同经度。地区统计使用近似经纬度窗口。版本、随机种子、重抽样长度及每项推断的源数据见相应日志与 CSV。

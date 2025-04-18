# Dòngchá

<img align="left" width="80" height="80" src="./img/dongcha_logo_circle.png">

The README file is useful for projects that are using __dongcha__ platform for AI/ML and augmented BI pipelines. It is designed to integrate data wrangling, mining, and visualization in to a single coherent project. Here we introduce ways for getting started with the platform framework. The [WIKI](/wiki) for comprehensive documentation on the dongcha methodology, functional components, and behaviour.

洞察 (Dòngchá): 洞 (dòng) – meaning "deep" or "penetrating" and 察 (chá) – meaning "to observe" or "to examine". Together, 洞察 (dòngchá) conveys the idea of keen observation leading to deep insight or clarity. It offers a Apache Spark and Python platform for data science.

__NOTE__: instructions and content is specific to Debian distros and was tested on Ubuntu 20.04.

## Table of Content
* [Starting a new project](#starting-a-new-project) - starting the _dongcha_ framework 
* [Test the newly set project](#test-the-new-project) - run pytest scripts to ensure _dongcha_ integrity
* [Updating dongcha in your project](#update-dongcha-from-remote-repo) - to pull the latest code from _dongcha_ repo and apply to your project submodule
* [Re-configuring an existing project](#reconfiguring-existing-project) - redoing the folders, init, and app.ini files
* [Description of the project artifacts](#about-the-post-setup-artifacts) - brief description of the essential framework files and folders

## Starting a New Project
1. Create an empty git repository with the a desired project name; e.g., __MyNewProj__ . 
   * Presupose that you have [git installed and initialized]([https://phoenixnap.com/kb/how-to-install-git-on-ubuntu](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)) on your computer.
1. Clone your _MyNewProj_ into a desired director location; for example
   * ```cd ~/all_dongcha_projects/```
   * ```git clone https://github.com/<my_git_user_name>/MyNewProj.git```
1. Move into the newly created project folder
   * ```cd ~/all_dongcha_projects/MyNewProj```
1. Now clone and initialize dongcha platform as a [submodule](https://github.blog/2016-02-01-working-with-submodules/)
   * ```git submodule add -b main https://github.com/waidyanatha/dongcha.git dongcha```
   * ```git submodule init```; will copy the mapping from the .gitmodules file into the local _./.git/config_ file
1. Navigate into the dongcha folder and run setup to initialize the project with AI/ML functional app classes
   * ```cd dongcha```
   * In the next command run the setup for dongcha separately and the apps separately
      - First run ```python3 -m 000_setup --app=dongcha --with_ini_files --init_proj_env```; it is important to use the _--with_ini_files_ and _--init_proj_env directive flag because it instructs _000_setup.py_ to build the _dongcha_ app and python __init.py__ and __app.ini__ files necessary for the seamless package integration; also setup the default poetry pyproject.toml and _.env_ files 
      - ```python3 -m 000_setup```; at the onset you would not have any _wrangler_, _mining_, and visuals code in the respective _modules_ folders; hence, you cannot build the python __init.py__ and __app.ini__ files. Without  the _--with_ini_files_ directive the process will simply generate the app folder structure and default _app.cfg_ file. 
   * You have now created your _MyNewProj_ with the _dongcha_ platform framework and can begin to start coding.
   * __Note__ you need to configure the _app.cfg_ in the _mining_,_wrangler_,and _visuals_ apps
      - each time you add new module packages; it needs to be added or removed from app.cfg
      - any other parameters, specific to the project must be changed.
1. Change back to the project director
   * ```cd ..``` or ```cd ~/all_dongcha_projects/MyNewProj```
1. Add the submodule and initialize
   * ```git add .gitmodules dongcha/```
   * ```git init```
1. Install dependencies with python _poetry_. 
   * The _pyproject.tom_ file would be created from the previous 000_setup.py step
   * ```poetry --version``` will confirm if _poetry_ dependency manager is installed
   * If required, follow the [poetry installation docs](https://python-poetry.org/docs/)
   * Activate the lock file with ```poetry lock```
   * Install dependencies with ```poetry install```
   * confirm installation and environment with ```poetry shell```; create a default shell with _(dongcha-py3.10)_
1. (Optional) Include a _README.md_ file, if not already
   * ```echo "# Welcome to MyNewProj" >> README.md```
1. Add and commit all newly created files and folders in _MyNewProj_
   * ```git add .```
   * ```git commit -m "added dongcha submudle and setup project"```
1. Push the submodule and new commits to the repo
   * ```git push origin main```
   * Check your github project in the browser; you will see a folder ___dongcha @ xxxxxxx___; where xxxxxxx is the last 7 digits from the _dongcha.git_ repo commit code 

## Test the new Project
Run __pytest__ by executing the command in your terminal prompt
* ```pytest```

## Update dongcha from remote repo
From time to time you will need to update the _dongcha_ submodule, in your project. 
1. change your directory to _MyNewProj_ folder
   * ```cd ~/all_dongcha_projects/MyNewProj```
1. fetch latest changes from _dongcha.git_ repository, and merge them into current _MyNewProj_ branch.
   * ```git submodule update --remote --merge```
1. update the repo in github:
   * ```git commit -s -am "updating dongcha submodule with latest"```
   * ```git push origin main```

## Reconfiguring existing project

When you add a new module package into the _mining_, _wrangler_, and _visuals_ app folders; as well as defining them in the _app.cfg_ file, the ___init___ and ___app.ini___ framework files need to be updated. For such simply run the _000_setup.py_
* ```cd ~/all_dongcha_projects/MyNewProj/dongcha``` navigate into the _dongcha_ folder
* ```python3 -m 000_setup --with_ini_files``` will re-configure all the apps
* Alternatively ```python3 -m 000_setup --app=wrangler,mining``` will only re-configure the specific apps


## About the Post Setup Artifacts

1. _Mining_ - Arificial Intelligence (AI) and Machine Learning (ML) analytical methods
1. _Wrangler_- for processing data extract, transform, and load automated pipelines
1. _Visuals_ - interactive dashboards with visual analytics for Business Intelligence (BI)
1. _utils.py_- contains a set of framework functions useful for all apps
1. _app.cfg_ - defines the app specific config section-wise key/value pairs
1. _Folders_ - each of the mining, wrangler, and visuals folders will contain a set of subfolders 
   * _dags_ - organizing airflow or other scheduler pipelines scripts
   * _data_ - specific parametric data and tmp files
   * _db_ - database scripts for creating the schema, tables, and initial data
   * _logs_ - log files created by each module package
   * _modules_ - managing the package functional class libraries
   * _notebooks_ - jupyter notebooks for developing and testing pipeline scripts
   * _tests_ - pytest scripts for applying unit & functional tests for any of the packages


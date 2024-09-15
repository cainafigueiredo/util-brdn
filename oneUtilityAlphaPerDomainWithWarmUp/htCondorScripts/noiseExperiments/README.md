# **Introdução**

Neste diretório, você irá encontrar tudo que é necessário para executar seus experimentos com dados de origem ruidosos em um cluster onde há suporte para a submissão de jobs através do HTCondor. Os jobs a serem submetidos são descritos através de um arquivo .sub. Ao especificar os jobs para os seus experimentos, você pode tomar como referência o arquivo de submissão `template.job`, adaptando-o para as suas necessidades. Antes, porém, você precisará realizar algumas etapas de preparação para os experimentos.

# **Preparando os experimentos**

1) O primeiro passo consiste em definir as variáveis de ambiente contidas em `.env`, apontando corretamente para os diretórios raízes do SRLearn e do projeto contendo os experimentos. Em seguida, execute o comando `source .env` em um terminal.

2) A seguir, gere o arquivo JSON que irá conter detalhes sobre as configurações dos experimentos que você deseja rodar. Para isso, execute o seguinte comando no mesmo terminal utilizado na etapa anterior:

```bash
$ python3 prepareExperiments.py --experimentsJSON <PATH_TO_RAW_JSON> --database <DATABASE_NAME> --beta <BETA_VALUE>
```

Se você estiver utilizando um ambiente virtual, lembre-se de ativá-lo antes de executar o comando acima. Nesse comando, <PATH_TO_RAW_JSON> é o caminho para o arquivo original contendo as especificações dos experimentos. Esse arquivo pode conter experimentos que você não deseja executar. Você consegue filtrar os experimentos através dos valores de <DATABASE_NAME> e <BETA_VALUE>, que representam o nome da base de dados utilizada nos experimentos a serem executados e a intensidade de balanceamento utilizada na estratégia de ponderação das instâncias, respectivamente. Após a execução desse comando, será gerado um arquivo no mesmo diretório cujo nome será `experiments-<DATABASE_NAME>-beta=<BETA_VALUE>.json`. Esse arquivo será referenciado posteriormente em um arquivo de submissão, conforme descrito a seguir. 

3) Crie um arquivo de submissão para os experimentos contidos em `experiments-<DATABASE_NAME>-beta=<BETA_VALUE>.json`. Você pode partir do `template.sub` e modificar apenas o argumento `--experimentsJSON` que é passado na chamada do script. Para isso, crie uma cópia do template fornecido.

```bash
$ cp ./template.sub ./experiments-<DATABASE_NAME>-beta=<BETA_VALUE>.sub
```

4) Tenha certeza de que os diretórios `./log`, `./error`, e `./output` existem. Caso não existem, então crie esses diretórios. 

```bash
$ mkdir log
$ mkdir error
$ mkdir output
```

# **Executando os experimentos**

Para executar os experimentos, você terá que submeter os jobs descritos em um arquivo de submissão.

```bash
$ condor_submit ./experiments-<DATABASE_NAME>-beta=<BETA_VALUE>.sub
```
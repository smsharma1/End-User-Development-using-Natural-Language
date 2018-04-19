from __future__ import unicode_literals
import json
import string



factionkb   =   open("../data/val_commands_discourse_tagger.json")

commands    =   json.load(factionkb)

fcommandparammatched = open("../data/val_commands_param.json","w+")


f = open("../data/nlphrases.tok.charniak.parse.dep")
depparser = f.read()
f.close()
i = 0
print(commands)
depparser = depparser.split("\n\n")
for command in commands:
    print(depparser[i])
    print(command["nl_command_statment"])
    command["param"] = []
    for par in depparser[i].split("\n"):
        if par.split("(")[0] in ["dobj","pobj"]:
            command["param"].append(par.split("(")[1].split(",")[1].split("-")[0])
    i+=1
    # print(command)
    # exit()
print(commands)
# exit()
json.dump(commands,fcommandparammatched,indent=4, sort_keys=True)
fcommandparammatched.close()

# print(depparser.split("\n\n")[0].split("\n")[0].split("("))

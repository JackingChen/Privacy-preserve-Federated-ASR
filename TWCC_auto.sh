TWCC_CLI_CMD=/home/jack/anaconda3/bin/twccli
#<USERNAME>：主機帳號

echo "1. 開啟instance"
# twccli mk ccs -gpu 1 -img pytorch-23.02-py3:latest -itype PyTorch -ptype c.super -wait
twccli mk ccs -gpu 1 -img pytorch-23.02-py3:FedLearn20230513 -itype 'Custom Image' -ptype c.super -wait

ID=$(twccli ls ccs | awk '/\|/ {print $2}' | awk 'NR==2')
info=$(twccli ls ccs $ID -gssh)
echo "ssh $info"
#設定SSH Without Password
#ssh-keygen -t rsa -n 4096
#ssh-copy-id u1157393@203-145-216-222.ccs.twcc.ai -p 50866
# passwd : BIIClab713isbest
ssh -o StrictHostKeyChecking=no $info



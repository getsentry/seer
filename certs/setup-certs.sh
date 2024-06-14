#!/usr/bin/env bash
set -xe

cd ca
cfssl genkey -initca ca-csr.json
cfssl genkey -initca ca-csr.json | cfssljson -bare ca
chmod a+r *.pem

cd ../server
cfssl gencert -ca=../ca/ca.pem -ca-key=../ca/ca-key.pem \
  -config=./server-config.json server.json | cfssljson -bare server
chmod a+r *.pem

mkdir -p /app/certs/client
cd ../client
cfssl gencert -ca=../ca/ca.pem -ca-key=../ca/ca-key.pem \
  -config=./client-config.json client.json | cfssljson -bare client
chmod a+r *.pem

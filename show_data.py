import h5py

# 파일 경로에 kakao-arena 에서 다운받은 train.chunk.01의 경로를 넣으셈
h = h5py.File("파일 경로", "r")
h = h['train']
# 키 전체값 출력
print(h.keys())
# 각각의 키값에 해당하는 데이터들을 보려면 h['product'][0] 이런식으로 치면 됨
# python3 인경우 한국어 데이터인경우 h['product'][0].decode('utf-8') 치면 됨
# 이 코드를 직접 돌리는 것보다 python script 실행해서 실시간으로 보면 편함

import lilac as ll

project_dir = '/workspace/lilac/bank-project'
server = ll.start_server(host='0.0.0.0', port=5432, project_dir=project_dir)

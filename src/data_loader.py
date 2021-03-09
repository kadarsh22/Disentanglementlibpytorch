def get_data_loader(config):
	if config['dataset'] == 'dsprites':
		from data.dsprites import DSprites
		data = DSprites(config)
		return data
	elif config['dataset'] == 'shapes_3d':
		from data.shapes_3d import shapes3d
		data = shapes3d(config)
		return data
	elif config['dataset'] == 'noisy_dsprites':
		raise NotImplementedError
	elif config['dataset'] == 'colored_dsprites':
		raise NotImplementedError
	elif config['dataset'] == 'celebA':
		raise NotImplementedError
	elif config['dataset'] == '':
		raise NotImplementedError
	else:
		raise NotImplementedError

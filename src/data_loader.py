def get_data_loader(config):
	if config['dataset'] == 'dsprites':
		from data.dsprites import DSprites
		data = DSprites(config)
		return data
	elif config['dataset'] == 'cdsprites':
		from data.CDsprties import CircularDSprites
		data = CircularDSprites(config)
		return data
	elif config['dataset'] == 'scream_dsprites':
		from data.dsprites import ScreamDSprites
		data = ScreamDSprites(config)
		return data
	elif config['dataset'] == 'noisy_dsprites':
		raise NotImplementedError
	elif config['dataset'] == 'celebA':
		raise NotImplementedError
	elif config['dataset'] == '':
		raise NotImplementedError
	else:
		raise NotImplementedError

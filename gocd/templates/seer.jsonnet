local seer = import './pipelines/seer.libsonnet';
local pipedream = import 'github.com/getsentry/gocd-jsonnet/libs/pipedream.libsonnet';

local pipedream_config = {
  name: 'seer',
  auto_deploy: false,
  exclude_regions: [
    'customer-1',
    'customer-2',
    'customer-3',
    'customer-6',
    'customer-7',
  ],
  materials: {
    seer_repo: {
      git: 'git@github.com:getsentry/seer.git',
      shallow_clone: true,
      branch: 'main',
      destination: 'seer',
    },
  },
  rollback: {
    material_name: 'seer_repo',
    stage: 'deploy-primary',
    elastic_profile_id: 'seer',
  },
};

pipedream.render(pipedream_config, seer)

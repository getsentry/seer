local gocdtasks = import 'github.com/getsentry/gocd-jsonnet/libs/gocd-tasks.libsonnet';

function(region) {
  materials: {
    seer_repo: {
      git: 'git@github.com:getsentry/seer.git',
      shallow_clone: true,
      branch: 'main',
      destination: 'seer',
    },
  },
  lock_behavior: 'unlockWhenFinished',
  stages: [
    {
      checks: {
        fetch_materials: true,
        environment_variables: {
          GITHUB_TOKEN: '{{SECRET:[devinfra-github][token]}}',
        },
        jobs: {
          check: {
            timeout: 1200,
            elastic_profile_id: 'seer',
            tasks: [
              gocdtasks.script(importstr '../bash/check-github.sh'),
              gocdtasks.script(importstr '../bash/check-cloudbuild.sh'),
            ],
          },
        },
      },
    },
    {
      'run-migrations': {
        environment_variables: {
          SENTRY_REGION: region,
        },
        jobs: {
          'run-migrations': {
            timeout: 1200,
            elastic_profile_id: 'seer',
            tasks: [
              gocdtasks.script(importstr '../bash/run-migrations.sh'),
            ],
          },
        },
      },
    },
    {
      'deploy-primary': {
        environment_variables: {
          SENTRY_REGION: region,
        },
        jobs: {
          'create-sentry-release': {
            timeout: 120,
            elastic_profile_id: 'seer',
            environment_variables: {
              SENTRY_AUTH_TOKEN: '{{SECRET:[devinfra-sentryio][token]}}',
            },
            tasks: [
              gocdtasks.script(importstr '../bash/create-sentry-release.sh'),
            ],
          },
          deploy: {
            timeout: 1200,
            elastic_profile_id: 'seer',
            tasks: [
              gocdtasks.script(importstr '../bash/deploy.sh'),
            ],
          },
        },
      },
    },
  ],
}

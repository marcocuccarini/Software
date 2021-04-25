
def benchmark(dsc,fname, learnL, learn2, results, useRocAuc=True):
  """ Legacy (non OO) """

  model_results = dict()
  cv_results = { "loss":[], "accuracy":[], "fbeta":[], "mcc":[], "rocauc":[] }

  learners = learnL + [learn2]

  for learner in learners:

    benchmark_type = 'test' if (learner == learn2) else 'valid'

    print('*'*30)
    print(benchmark_type + ' metrics')

    interp = ClassificationInterpretation.from_learner(learner)
  
    interp.plot_confusion_matrix(figsize=(10, 10))
    plt.show()

    interp.print_classification_report()

    if (benchmark_type == 'valid'):

      cv_metrics = learner.validate()
      
      cv_metrics = iter(cv_metrics)

      cv_results['loss'].append(next(cv_metrics))

      cv_results['accuracy'].append(next(cv_metrics))

      cv_results['fbeta'].append(next(cv_metrics))

      cv_results['mcc'].append(next(cv_metrics))

      if (useRocAuc):
        cv_results['rocauc'].append(next(cv_metrics))

      if (len(cv_results['loss']) == len(learners) - 1):
        model_results[benchmark_type] = dict()

        model_results[benchmark_type]['loss'] = sum(cv_results['loss']) / len(cv_results['loss']) 

        model_results[benchmark_type]['accuracy'] = sum(cv_results['accuracy']) / len(cv_results['accuracy']) 

        model_results[benchmark_type]['fbeta'] = sum(cv_results['fbeta']) / len(cv_results['fbeta']) 

        model_results[benchmark_type]['mcc'] = sum(cv_results['mcc']) / len(cv_results['mcc']) 

        if (useRocAuc):
          model_results[benchmark_type]['rocauc'] = sum(cv_results['rocauc']) / len(cv_results['rocauc']) 

    else:
      model_metrics = learner.validate()
      
      model_metrics = iter(model_metrics)

      model_results[benchmark_type] = dict()

      model_results[benchmark_type]['loss'] = next(model_metrics)

      model_results[benchmark_type]['accuracy'] = next(model_metrics)

      model_results[benchmark_type]['fbeta'] = next(model_metrics)

      model_results[benchmark_type]['mcc'] = next(model_metrics)

      if (useRocAuc):
        model_results[benchmark_type]['rocauc'] = next(model_metrics)

    print('\n'*2)
  
  results[fname] = model_results
  
  pprint(model_results)
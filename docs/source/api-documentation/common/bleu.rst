BLEU
====================================================

.. py:module:: neural_compressor.experimental.metric.bleu


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neural_compressor.experimental.metric.bleu.BLEU

.. py:class:: BLEU

   Bases: :py:obj:`object`

   Computes the BLEU (Bilingual Evaluation Understudy) score.

   BLEU is an algorithm for evaluating the quality of text which has
   been machine-translated from one natural language to another.
   This implementent approximate the BLEU score since we do not
   glue word pieces or decode the ids and tokenize the output.
   By default, we use ngram order of 4 and use brevity penalty.
   Also, this does not have beam search.

   .. attribute:: predictions

      List of translations to score.

   .. attribute:: labels

      List of the reference corresponding to the prediction result.

   .. py:method:: reset() -> None

      Clear the predictions and labels in the cache.


   .. py:method:: update(prediction: Sequence[str], label: Sequence[str]) -> None

      Add the prediction and label.

      :param prediction: The prediction result.
      :param label: The reference corresponding to the prediction result.

      :raises ValueError: An error occurred when the length of the prediction
      :raises and label are different.:


   .. py:method:: result() -> float

      Compute the BLEU score.

      :returns: The approximate BLEU score.
      :rtype: bleu_score



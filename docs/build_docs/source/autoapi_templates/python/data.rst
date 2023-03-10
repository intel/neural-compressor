{% if obj.display %}
.. py:{{ obj.type }}:: {{ obj.name }}
   {%- if obj.annotation is not none %}

   :type: {%- if obj.annotation %} {{ obj.annotation }}{%- endif %}

   {%- endif %}

   {%- if obj.value is not none %}

   :value: {% if obj.value is string and obj.value.splitlines()|count > 1 -%}
                Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: python

        """{{ obj.value|indent(width=8,blank=true) }}"""

    .. raw:: html

        </details>

            {%- else -%}
              {%- if obj.value is string -%}
                {{ "%r" % obj.value|string|truncate(100) }}
              {%- else -%}
                {{ obj.value|string|truncate(100) }}
              {%- endif -%}
            {%- endif %}
   {%- endif %}


   {{ obj.docstring|indent(3) }}
{% endif %}
from django import template

register = template.Library()


@register.simple_tag
def max_by(list, attribute):
    return max(list, key=lambda x: getattr(x, attribute))


@register.simple_tag
def min_by(list, attribute):
    return min(list, key=lambda x: getattr(x, attribute))
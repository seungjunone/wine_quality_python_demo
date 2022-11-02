with red as (

    select 
        * ,
        'red' as target
    from {{ ref('stg_winequality_red') }}

),

white as (

    select 
        *,
        'white' as target 
    from {{ ref('stg_winequality_white') }}

)

SELECT * FROM red

UNION ALL

SELECT * FROM white
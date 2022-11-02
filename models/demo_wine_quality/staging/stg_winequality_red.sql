with source as (

    select * from {{ ref('raw_winequality_red') }}

)

select * from source